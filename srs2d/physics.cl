#include <pyopencl-ranluxcl.cl>

#define PI 3.141592653589793115997963
#define DEG2RAD 0.017453292519943295

#define ROBOT_BODY_RADIUS           0.06
#define WHEELS_MAX_ANGULAR_SPEED    12.565
#define WHEELS_DISTANCE             0.0825
#define WHEELS_RADIUS               0.02

#define IR_RADIUS       0.025
#define CAMERA_RADIUS   0.35
#define CAMERA_ANGLE    72

#define LED_SIZE        0.01

#define NUM_INPUTS      13
#define NUM_OUTPUTS      4

#define IN_camera0       0
#define IN_camera1       1
#define IN_camera2       2
#define IN_camera3       3
#define IN_proximity0    4
#define IN_proximity1    5
#define IN_proximity2    6
#define IN_proximity3    7
#define IN_proximity4    8
#define IN_proximity5    9
#define IN_proximity6   10
#define IN_proximity7   11
#define IN_ground0      12

#define OUT_wheels0      0
#define OUT_wheels1      1
#define OUT_front_led0   2
#define OUT_rear_led0    3

typedef struct {
    float2 pos;
    float2 rot;
} transform_t;

typedef struct {
    transform_t transform;
    float2 wheels_angular_speed;
    int front_led;
    int rear_led;
    int collision_count;
} robot_t;

typedef struct {
    float2 center;
    float radius;
} target_area_t;

float2 rot_mul_vec(float2 r, float2 v)
{
    float2 res = { r.s1 * v.x - r.s0 * v.y,
                   r.s0 * v.x + r.s1 * v.y };
    return res;
}

float2 transform_mul_vec(transform_t t, float2 v)
{
    float2 res = { (t.rot.s1 * v.x - t.rot.s0 * v.y) + t.pos.x,
                   (t.rot.s0 * v.x + t.rot.s1 * v.y) + t.pos.y };
    return res;
}

float dist_sq(float2 p1, float2 p2)
{
    return pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2);
}

float dist(float2 p1, float2 p2)
{
    return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}

float raycast(__global robot_t *robots, float2 p1, float2 p2)
{
    int self = get_global_id(0);
    int num_robots = get_global_size(0);
    unsigned int i;
    float ray_dist = dist(p1, p2);
    float min = ray_dist;

    float2 d = {p2.x - p1.x, p2.y - p1.y}; // vector of ray, from start to end
    float a = (d.x * d.x) + (d.y * d.y);   // d dot d

    for (i = 0; i < num_robots; i++)
    {
        if (i == self)
            continue;

        float dist_robot_sq = dist_sq(p1, robots[i].transform.pos);
        if (dist_robot_sq < pow(ray_dist + ROBOT_BODY_RADIUS, 2))
        {
            float2 f = {p1.x - robots[i].transform.pos.x,
                        p1.y - robots[i].transform.pos.y }; // vector from center sphere to ray start

            float b = 2 * ((f.x * d.x) + (f.y * d.y)); // f dot d
            float c = ((f.x * f.x) + (f.y * f.y)) - pow(ROBOT_BODY_RADIUS, 2); // f dot f - r**2

            float discriminant = (b*b - 4 * a *c);

            if (discriminant < 0)
                continue;

            discriminant = sqrt(discriminant);
            float x1 = (-b - discriminant) / (2*a);
            float x2 = (-b + discriminant) / (2*a);

            if ((x1 >= 0 && x1 <= 1) || (x2 >= 0 && x2 <= 1))
            {
                float dist_obstacle = sqrt(dist_robot_sq);
                if (dist_obstacle < min)
                    min = dist_obstacle;
            }
        }
    }

    return min;
}

__kernel void size_of_transform_t(__global int *result)
{
    *result = (int) sizeof(transform_t);
}

__kernel void size_of_robot_t(__global int *result)
{
    *result = (int) sizeof(robot_t);
}

__kernel void size_of_target_area_t(__global int *result)
{
    *result = (int) sizeof(target_area_t);
}

__kernel void init_ranluxcl(uint ins, __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_initialization(ins, ranluxcltab);
}

__kernel void init_robots(__global ranluxcl_state_t *ranluxcltab, __global robot_t *robots)
{
    int gid = get_global_id(0);

    // random_position
    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
    float4 random = ranluxcl32(&ranluxclstate);
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

    robots[gid].transform.pos.x = random.s0 * 2 - 1;
    robots[gid].transform.pos.y = random.s1 * 2 - 1;
    robots[gid].transform.rot.s0 = 0;
    robots[gid].transform.rot.s1 = 1;
    robots[gid].wheels_angular_speed.s0 = 0;
    robots[gid].wheels_angular_speed.s1 = 0;
    robots[gid].front_led = 0;
    robots[gid].rear_led = 0;
    robots[gid].collision_count = 0;
}

__kernel void init_target_areas(__global target_area_t *target_areas, const float targets_distance)
{
    int gid = get_global_id(0);

    float x = sqrt(pow((targets_distance / 2.0), 2) / 2.0);

    if (gid == 0)
    {
        target_areas[gid].center.x = -x;
        target_areas[gid].center.y = x;
    }
    else
    {
        target_areas[gid].center.x = x;
        target_areas[gid].center.y = -x;
    }

    target_areas[gid].radius = 0.27;
}

__kernel void step_actuators(__global robot_t *robots, __global const float *outputs)
{
    int gid = get_global_id(0);

    robots[gid].wheels_angular_speed.s0 = outputs[gid*NUM_OUTPUTS+OUT_wheels0] * WHEELS_MAX_ANGULAR_SPEED;
    robots[gid].wheels_angular_speed.s1 = outputs[gid*NUM_OUTPUTS+OUT_wheels1] * WHEELS_MAX_ANGULAR_SPEED;
    robots[gid].front_led = outputs[gid*NUM_OUTPUTS+OUT_front_led0];
    robots[gid].rear_led = outputs[gid*NUM_OUTPUTS+OUT_rear_led0];
}

__kernel void step_dynamics(const float time_step, __global robot_t *robots)
{
    int gid = get_global_id(0);

    float2 wls; // wheels linear speed
    wls.s0 = robots[gid].wheels_angular_speed.s0 * WHEELS_RADIUS;
    wls.s1 = robots[gid].wheels_angular_speed.s1 * WHEELS_RADIUS;

    float2 wlv = {wls.s0 + wls.s1, 0}; // wheels linear velocity

    float2 linear_velocity = rot_mul_vec(robots[gid].transform.rot, wlv);

    float angular_speed = atan2(wls.s0 - wls.s1, (float) WHEELS_DISTANCE / 2);

    robots[gid].transform.pos.x += linear_velocity.x * time_step;
    robots[gid].transform.pos.y += linear_velocity.y * time_step;

    float angle = atan2(robots[gid].transform.rot.s0, robots[gid].transform.rot.s1);
    angle += angular_speed * time_step;
    robots[gid].transform.rot.s0 = sin(angle);
    robots[gid].transform.rot.s1 = cos(angle);
}

__kernel void step_sensors(__global ranluxcl_state_t *ranluxcltab, __global robot_t *robots,
    __global target_area_t *target_areas, __global float *inputs, __global float *outputs)
{
    int gid = get_global_id(0);
    int num_robots = get_global_size(0);
    unsigned int i;

    for (i = 0; i < NUM_INPUTS; i++)
        inputs[gid*NUM_INPUTS+i] = 0.0;

    for (i = 0; i < num_robots; i++)
    {
        float dist_robot_sq = dist_sq(robots[gid].transform.pos, robots[i].transform.pos);

        if (i > gid)
        {
            if (dist_robot_sq < pow(2 * ROBOT_BODY_RADIUS, 2))
            {
                // random_position
                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
                float4 random = ranluxcl32(&ranluxclstate);
                ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

                robots[gid].transform.pos.x = random.s0 * 4 - 2;
                robots[gid].transform.pos.y = random.s1 * 4 - 2;
                robots[gid].transform.rot.s0 = random.s2 * 2 - 1;
                robots[gid].transform.rot.s1 = random.s3 * 2 - 1;
            }
        }

        if (dist_robot_sq < pow(2 * ROBOT_BODY_RADIUS + IR_RADIUS, 2))
        {
            float s = robots[i].transform.pos.y - robots[gid].transform.pos.y;
            float c = robots[i].transform.pos.x - robots[gid].transform.pos.x;
            float a = atan2(s, c) + atan2(robots[i].transform.pos.s0, robots[i].transform.pos.s1);
            int idx = (int) floor(a / (2*PI/8)) % 8;
            inputs[gid*NUM_INPUTS+IN_proximity0+idx] = sqrt(dist_robot_sq) - 2*ROBOT_BODY_RADIUS;
        }

        if (dist_robot_sq < pow(CAMERA_RADIUS + ROBOT_BODY_RADIUS + LED_SIZE, 2))
        {
            float2 front = {(ROBOT_BODY_RADIUS + LED_SIZE), 0};
            float2 rear = {-(ROBOT_BODY_RADIUS + LED_SIZE), 0};

            float2 orig = robots[gid].transform.pos;
            float angle_robot = atan2(robots[gid].transform.rot.s0, robots[gid].transform.rot.s1);

            if (outputs[i*NUM_OUTPUTS+OUT_front_led0] > 0.5)
            {
                float2 dest = transform_mul_vec(robots[i].transform, front);

                if (dist_sq(orig, dest) < pow(CAMERA_RADIUS, 2))
                {
                    float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= ((CAMERA_ANGLE/2)*DEG2RAD)) )
                    {
                        inputs[gid*NUM_INPUTS+IN_camera0] = 1 - (raycast(robots, orig, dest) / CAMERA_RADIUS);
                    }
                    else if ( (angle_dest > angle_robot) &&
                              ((angle_dest-angle_robot) <= ((CAMERA_ANGLE/2)*DEG2RAD)) )
                    {
                        inputs[gid*NUM_INPUTS+IN_camera1] = 1 - (raycast(robots, orig, dest) / CAMERA_RADIUS);
                    }
                }
            }

            if (outputs[i*NUM_OUTPUTS+OUT_rear_led0] > 0.5)
            {
                float2 dest = transform_mul_vec(robots[i].transform, rear);

                if (dist_sq(orig, dest) < pow(CAMERA_RADIUS, 2))
                {
                    float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= ((CAMERA_ANGLE/2)*DEG2RAD)) )
                    {
                        inputs[gid*NUM_INPUTS+IN_camera2] = 1 - (raycast(robots, orig, dest) / CAMERA_RADIUS);
                    }
                    else if ( (angle_dest > angle_robot) &&
                         ((angle_dest-angle_robot) <= ((CAMERA_ANGLE/2)*DEG2RAD)) )
                    {
                        inputs[gid*NUM_INPUTS+IN_camera3] = 1 - (raycast(robots, orig, dest) / CAMERA_RADIUS);
                    }
                }
            }
        }
    }

    for (i = 0; i < 2; i++)
    {
        float dist_ta_sq = dist_sq(robots[gid].transform.pos, target_areas[i].center);

        if (dist_ta_sq < pow(target_areas[i].radius, 2))
            inputs[gid*NUM_INPUTS+IN_ground0] = 1;
        else
            inputs[gid*NUM_INPUTS+IN_ground0] = 0;
    }
}

__kernel void get_transform_matrices(__global robot_t *robots, __global float4 *transforms)
{
    int gid = get_global_id(0);

    transforms[gid].s0 = robots[gid].transform.pos.x;
    transforms[gid].s1 = robots[gid].transform.pos.y;
    transforms[gid].s2 = robots[gid].transform.rot.s0;
    transforms[gid].s3 = robots[gid].transform.rot.s1;
}