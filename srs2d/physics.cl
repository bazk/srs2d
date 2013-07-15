#define ROBOT_BODY_RADIUS           0.06
#define WHEELS_MAX_ANGULAR_SPEED    12.565
#define WHEELS_DISTANCE             0.0825
#define WHEELS_RADIUS               0.02

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

__kernel void init_robots(__global robot_t *robots)
{
    int gid = get_global_id(0);

    robots[gid].transform.pos.x = gid * 0.15;
    robots[gid].transform.pos.y = 0;
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

    float2 linear_velocity;
    linear_velocity.x = robots[gid].transform.rot.s0 * (wls.s0 + wls.s1);
    linear_velocity.y = robots[gid].transform.rot.s1 * (wls.s0 + wls.s1);

    float angular_speed = (wls.s0 - wls.s1) / (WHEELS_DISTANCE / 2);

    robots[gid].transform.pos.x += linear_velocity.x * time_step;
    robots[gid].transform.pos.y += linear_velocity.y * time_step;

    float angle = atan2(robots[gid].transform.rot.s0, robots[gid].transform.rot.s1);
    angle += angular_speed * time_step;
    robots[gid].transform.rot.s0 = sin(angle);
    robots[gid].transform.rot.s1 = cos(angle);
}

__kernel void step_sensors(__global robot_t *robots, __global const float *inputs)
{
    int gid = get_global_id(0);

    // TODO
}