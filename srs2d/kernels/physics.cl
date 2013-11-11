#include <pyopencl-ranluxcl.cl>

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define ROBOT_BODY_RADIUS           0.035
#define WHEELS_MAX_ANGULAR_SPEED    2.05
#define WHEELS_DISTANCE             0.055
#define WHEELS_RADIUS               0.02
#define IR_RADIUS                   0.025
#define CAMERA_RADIUS               0.35
#define CAMERA_ANGLE                1.2566370614359172 // 72 degrees
#define LED_PROTUBERANCE            0.007
#define TARGET_AREAS_RADIUS         0.32

#define ARENA_HEIGHT    2.5
#define ARENA_WIDTH_MIN 2.5
#define ARENA_WIDTH_MAX 2.9

#define NUM_SENSORS    13
#define NUM_ACTUATORS   4
#define NUM_HIDDEN      3

#define IN_proximity0   1
#define IN_proximity1   2
#define IN_proximity2   4
#define IN_proximity3   8
#define IN_proximity4   16
#define IN_proximity5   32
#define IN_proximity6   64
#define IN_proximity7   128
#define IN_camera0      256
#define IN_camera1      512
#define IN_camera2      1024
#define IN_camera3      2048
#define IN_ground       4096

#define OUT_wheels0     0
#define OUT_wheels1     1
#define OUT_front_led   2
#define OUT_rear_led    3

#define HID_hidden0     0
#define HID_hidden1     1
#define HID_hidden2     2

typedef struct {
    float sin;
    float cos;
} rotation_t;

typedef struct {
    float2 pos;
    rotation_t rot;
} transform_t;

typedef struct {
    transform_t transform;
    float2 wheels_angular_speed;
    int front_led;
    int rear_led;
    int collision;
    float energy;
    float fitness;
    int last_target_area;

    unsigned int sensors;
    float actuators[NUM_ACTUATORS];
    float hidden[NUM_HIDDEN];
} robot_t;

typedef struct {
    float2 center;
    float radius;
} target_area_t;

typedef struct {
    robot_t robots[ROBOTS_PER_WORLD];

    float k;

    float arena_height;
    float arena_width;

    float targets_distance;

    target_area_t target_areas[2];

    // ANN parameters
    float weights[NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)];
    float bias[NUM_ACTUATORS];

    float weights_hidden[NUM_HIDDEN*NUM_SENSORS];
    float bias_hidden[NUM_HIDDEN];
    float timec_hidden[NUM_HIDDEN];
} world_t;

__kernel void size_of_world_t(__global int *result)
{
    *result = (int) sizeof(world_t);
}

float2 rot_mul_vec(rotation_t r, float2 v)
{
    float2 res = { r.cos * v.x - r.sin * v.y,
                   r.sin * v.x + r.cos * v.y };
    return res;
}

float2 transform_mul_vec(transform_t t, float2 v)
{
    float2 res = { (t.rot.cos * v.x - t.rot.sin * v.y) + t.pos.x,
                   (t.rot.sin * v.x + t.rot.cos * v.y) + t.pos.y };
    return res;
}

unsigned int uint_exp2(unsigned int n)
{

    return floor(exp2((float) n));
}

float sigmoid(float z)
{
    return 1 / (1 + exp(-z));
}

__kernel void test_sigmoid(__global float *z)
{
    float2 p1 = {0,0};
    float2 p2 = {1,0};

    *z = distance(p1, p2);
}

bool raycast(__global world_t *worlds, float2 p1, float2 p2)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int otherid;

    float2 ray = {p2.x - p1.x, p2.y - p1.y};
    float ray_length = length(ray);
    float2 ray_unit = {ray.x / ray_length, ray.y / ray_length};

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        if (rid == otherid)
            continue;

        float dist = distance(p1, worlds[wid].robots[otherid].transform.pos);

        if (dist < ray_length + ROBOT_BODY_RADIUS)
        {
            float2 v1 = { worlds[wid].robots[otherid].transform.pos.x - p1.x,
                          worlds[wid].robots[otherid].transform.pos.y - p1.y };

            float proj = v1.x * ray_unit.x + v1.y * ray_unit.y;

            if ((proj > 0) && (proj < ray_length)) {
                float2 d = {ray_unit.x * proj, ray_unit.y * proj};
                d.x += p1.x;
                d.y += p1.y;

                if (distance(d, worlds[wid].robots[otherid].transform.pos) < ROBOT_BODY_RADIUS)
                    return true;
            }
        }
    }

    return false;
}

int _raycast(__global world_t *worlds)
{
    float2 front = {(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};

    float2 orig = worlds[0].robots[0].transform.pos;
    float angle_robot = atan2(worlds[0].robots[0].transform.rot.sin, worlds[0].robots[0].transform.rot.cos);

    float2 dest = transform_mul_vec(worlds[0].robots[2].transform, front);
    float d = distance(orig, dest);

    if (d < CAMERA_RADIUS)
    {
        float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

        if ( (angle_robot > angle_dest) &&
             ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
        {
            if (!raycast(worlds, orig, dest))
                return 1;
        }
        else if ( (angle_dest > angle_robot) &&
                  ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
        {
            if (!raycast(worlds, orig, dest))
                return 2;
        }
    }

    return 0;
}

__kernel void test_raycast(__global world_t *worlds, __global int *result)
{
    worlds[0].robots[0].transform.pos.x = 0;
    worlds[0].robots[0].transform.pos.y = 0;
    worlds[0].robots[0].transform.rot.sin = 0;
    worlds[0].robots[0].transform.rot.cos = 1;

    worlds[0].robots[1].transform.pos.x = .15;
    worlds[0].robots[1].transform.pos.y = -.10;
    worlds[0].robots[1].transform.rot.sin = 0;
    worlds[0].robots[1].transform.rot.cos = 1;

    worlds[0].robots[2].transform.pos.x = .20;
    worlds[0].robots[2].transform.pos.y = .10;
    worlds[0].robots[2].transform.rot.sin = -1;
    worlds[0].robots[2].transform.rot.cos = 0;

    result[0] = _raycast(worlds);

    worlds[0].robots[0].transform.pos.x = 0;
    worlds[0].robots[0].transform.pos.y = 0;
    worlds[0].robots[0].transform.rot.sin = 0;
    worlds[0].robots[0].transform.rot.cos = 1;

    worlds[0].robots[1].transform.pos.x = .10;
    worlds[0].robots[1].transform.pos.y = .03;
    worlds[0].robots[1].transform.rot.sin = 0;
    worlds[0].robots[1].transform.rot.cos = 1;

    worlds[0].robots[2].transform.pos.x = .20;
    worlds[0].robots[2].transform.pos.y = .10;
    worlds[0].robots[2].transform.rot.sin = -1;
    worlds[0].robots[2].transform.rot.cos = 0;

    result[1] = _raycast(worlds);

    worlds[0].robots[0].transform.pos.x = 0;
    worlds[0].robots[0].transform.pos.y = 0;
    worlds[0].robots[0].transform.rot.sin = 0;
    worlds[0].robots[0].transform.rot.cos = 1;

    worlds[0].robots[1].transform.pos.x = .15,
    worlds[0].robots[1].transform.pos.y = .03;
    worlds[0].robots[1].transform.rot.sin = 0;
    worlds[0].robots[1].transform.rot.cos = 1;

    worlds[0].robots[2].transform.pos.x = .05;
    worlds[0].robots[2].transform.pos.y = .20;
    worlds[0].robots[2].transform.rot.sin = -1;
    worlds[0].robots[2].transform.rot.cos = 0;

    result[2] = _raycast(worlds);

    worlds[0].robots[0].transform.pos.x = 0;
    worlds[0].robots[0].transform.pos.y = 0;
    worlds[0].robots[0].transform.rot.sin = 1;
    worlds[0].robots[0].transform.rot.cos = 0;

    worlds[0].robots[1].transform.pos.x = -.04,
    worlds[0].robots[1].transform.pos.y = -.10;
    worlds[0].robots[1].transform.rot.sin = 0;
    worlds[0].robots[1].transform.rot.cos = 1;

    worlds[0].robots[2].transform.pos.x = .05;
    worlds[0].robots[2].transform.pos.y = .20;
    worlds[0].robots[2].transform.rot.sin = -1;
    worlds[0].robots[2].transform.rot.cos = 0;

    result[3] = _raycast(worlds);
}

__kernel void set_random_position(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    int otherid, i, collision = 1;

    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

    while (collision == 1)
    {
        float4 random = ranluxcl32(&ranluxclstate);

        float max_x = (worlds[wid].arena_width / 2) - ROBOT_BODY_RADIUS;
        float max_y = (worlds[wid].arena_height / 2) - ROBOT_BODY_RADIUS;
        worlds[wid].robots[rid].transform.pos.x = (random.s0 * 2 * max_x) - max_x;
        worlds[wid].robots[rid].transform.pos.y = (random.s1 * 2 * max_y) - max_y;
        worlds[wid].robots[rid].transform.rot.sin = random.s2 * 2 - 1;
        worlds[wid].robots[rid].transform.rot.cos = random.s3 * 2 - 1;

        collision = 0;

        // check for collision with other robots
        for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
        {
            if (rid != otherid)
            {
                float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].robots[otherid].transform.pos);

                if (dist < 2*ROBOT_BODY_RADIUS)
                {
                    collision = 1;
                    break;
                }
            }
        }

        // check for "collision" with target areas
        for (i = 0; i < 2; i++)
        {
            float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].target_areas[i].center);

            if (dist < worlds[wid].target_areas[i].radius)
            {
                collision = 1;
                break;
            }
        }
    }

    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
}

__kernel void init_ranluxcl(uint seed, __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_initialization(seed, ranluxcltab);
}

__kernel void init_worlds(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float targets_distance)
{
    int wid = get_global_id(0);

    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
    float4 random = ranluxcl32(&ranluxclstate);
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

    worlds[wid].targets_distance = targets_distance;

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    worlds[wid].k = (targets_distance / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

    worlds[wid].arena_height = ARENA_HEIGHT;
    worlds[wid].arena_width = ARENA_WIDTH_MIN + random.s0 * (ARENA_WIDTH_MAX - ARENA_WIDTH_MIN);

    float x = sqrt(pow((targets_distance / 2.0), 2) / 2.0);

    worlds[wid].target_areas[0].center.x = -x;
    worlds[wid].target_areas[0].center.y = x;
    worlds[wid].target_areas[1].center.x = x;
    worlds[wid].target_areas[1].center.y = -x;

    worlds[wid].target_areas[0].radius = TARGET_AREAS_RADIUS;
    worlds[wid].target_areas[1].radius = TARGET_AREAS_RADIUS;
}

__kernel void init_robots(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int i;

    set_random_position(ranluxcltab, worlds);
    worlds[wid].robots[rid].wheels_angular_speed.s0 = 0;
    worlds[wid].robots[rid].wheels_angular_speed.s1 = 0;
    worlds[wid].robots[rid].front_led = 0;
    worlds[wid].robots[rid].rear_led = 0;
    worlds[wid].robots[rid].collision = 0;
    worlds[wid].robots[rid].energy = 2;
    worlds[wid].robots[rid].fitness = 0.0;
    worlds[wid].robots[rid].last_target_area = -1;

    worlds[wid].robots[rid].sensors = 0;

    for (i=0; i<NUM_ACTUATORS; i++)
        worlds[wid].robots[rid].actuators[i] = 0;

    for (i=0; i<NUM_HIDDEN; i++)
        worlds[wid].robots[rid].hidden[i] = 0;
}

__kernel void set_ann_parameters(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds,
    __global float *weights, __global float *bias, __global float *weights_hidden,
    __global float *bias_hidden, __global float *timec_hidden)
{
    int wid = get_global_id(0);

    unsigned int i, j;

    for (i=0; i<NUM_ACTUATORS; i++)
    {
        for (j=0; j<(NUM_SENSORS+NUM_HIDDEN); j++)
            worlds[wid].weights[i*(NUM_SENSORS+NUM_HIDDEN)+j] = weights[i*(NUM_SENSORS+NUM_HIDDEN)+j];

        worlds[wid].bias[i] = bias[i];
    }

    for (i=0; i<NUM_HIDDEN; i++)
    {
        for (j=0; j<NUM_SENSORS; j++)
            worlds[wid].weights_hidden[i*NUM_SENSORS+j] = weights_hidden[i*NUM_SENSORS+j];

        worlds[wid].bias_hidden[i] = bias_hidden[i];
        worlds[wid].timec_hidden[i] = timec_hidden[i];
    }
}

__kernel void step_controllers(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int s,h,a;
    float aux;

    for (h=0; h<NUM_HIDDEN; h++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += worlds[wid].weights_hidden[h*NUM_SENSORS+s] *
                  ( ((worlds[wid].robots[rid].sensors & uint_exp2(s)) == 0) ? 0 : 1 );

        aux += worlds[wid].bias_hidden[h];

        worlds[wid].robots[rid].hidden[h] = worlds[wid].timec_hidden[h] * worlds[wid].robots[rid].hidden[h] +
              (1 - worlds[wid].timec_hidden[h]) * sigmoid(aux);
    }

    for (a=0; a<NUM_ACTUATORS; a++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += worlds[wid].weights[a*(NUM_SENSORS+NUM_HIDDEN)+s] * ( ((worlds[wid].robots[rid].sensors & uint_exp2(s)) == 0) ? 0 : 1 );

        for (h=0; h<NUM_HIDDEN; h++)
            aux += worlds[wid].weights[a*(NUM_SENSORS+NUM_HIDDEN)+NUM_SENSORS+h] * worlds[wid].robots[rid].hidden[h];

        aux += worlds[wid].bias[a];

        worlds[wid].robots[rid].actuators[a] = sigmoid(aux);
    }
}

__kernel void step_actuators(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    worlds[wid].robots[rid].wheels_angular_speed.s0 = worlds[wid].robots[rid].actuators[OUT_wheels0] * WHEELS_MAX_ANGULAR_SPEED;
    worlds[wid].robots[rid].wheels_angular_speed.s1 = worlds[wid].robots[rid].actuators[OUT_wheels1] * WHEELS_MAX_ANGULAR_SPEED;
    worlds[wid].robots[rid].front_led = (worlds[wid].robots[rid].actuators[OUT_front_led] > 0.5) ? 1 : 0;
    worlds[wid].robots[rid].rear_led = (worlds[wid].robots[rid].actuators[OUT_rear_led] > 0.5) ? 1 : 0;
}

__kernel void step_dynamics(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    if (worlds[wid].robots[rid].collision != 0) {
        set_random_position(ranluxcltab, worlds);
        worlds[wid].robots[rid].collision = 0;
        worlds[wid].robots[rid].wheels_angular_speed.s0 = 0;
        worlds[wid].robots[rid].wheels_angular_speed.s1 = 0;
        worlds[wid].robots[rid].front_led = 0;
        worlds[wid].robots[rid].rear_led = 0;
        worlds[wid].robots[rid].energy = 2;
        worlds[wid].robots[rid].fitness = 0.0;
        worlds[wid].robots[rid].last_target_area = -1;
        return;
    }

    float2 wls; // wheels linear speed
    wls.s0 = worlds[wid].robots[rid].wheels_angular_speed.s0 * WHEELS_RADIUS;
    wls.s1 = worlds[wid].robots[rid].wheels_angular_speed.s1 * WHEELS_RADIUS;

    float2 wlv = {wls.s0 + wls.s1, 0}; // wheels linear velocity

    float2 linear_velocity = rot_mul_vec(worlds[wid].robots[rid].transform.rot, wlv);

    float angular_speed = atan2(wls.s0 - wls.s1, (float) WHEELS_DISTANCE / 2);

    worlds[wid].robots[rid].transform.pos.x += linear_velocity.x * TIME_STEP;
    worlds[wid].robots[rid].transform.pos.y += linear_velocity.y * TIME_STEP;

    float angle = atan2(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);
    angle += angular_speed * TIME_STEP;
    worlds[wid].robots[rid].transform.rot.sin = sin(angle);
    worlds[wid].robots[rid].transform.rot.cos = cos(angle);
}

__kernel void step_sensors(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int i, otherid;

    worlds[wid].robots[rid].sensors = 0;

    if ( ((worlds[wid].robots[rid].transform.pos.x+ROBOT_BODY_RADIUS) > (worlds[wid].arena_width/2)) ||
         ((worlds[wid].robots[rid].transform.pos.x-ROBOT_BODY_RADIUS) < (-worlds[wid].arena_width/2)) ||
         ((worlds[wid].robots[rid].transform.pos.y+ROBOT_BODY_RADIUS) > (worlds[wid].arena_height/2)) ||
         ((worlds[wid].robots[rid].transform.pos.y-ROBOT_BODY_RADIUS) < (-worlds[wid].arena_height/2)) )
    {
        worlds[wid].robots[rid].collision = 1;
        return;
    }

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        if (rid == otherid)
            continue;

        float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].robots[otherid].transform.pos);

        // if ((dist < 2*ROBOT_BODY_RADIUS) && (otherid > rid)) {
        if (dist < 2*ROBOT_BODY_RADIUS) {
            worlds[wid].robots[rid].collision = 1;
            return;
        }

        if (dist < 2*ROBOT_BODY_RADIUS+IR_RADIUS)
        {
            float s = worlds[wid].robots[otherid].transform.pos.y - worlds[wid].robots[rid].transform.pos.y;
            float c = worlds[wid].robots[otherid].transform.pos.x - worlds[wid].robots[rid].transform.pos.x;

            float a = atan2(s, c) - atan2(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);
            if (a < 0)
                a += 2*M_PI;

            int idx = (int) floor(fabs(a) / (2*M_PI/8)) % 8;
            worlds[wid].robots[rid].sensors |= uint_exp2(idx);
        }

        if (dist < CAMERA_RADIUS + ROBOT_BODY_RADIUS + LED_PROTUBERANCE)
        {
            float2 front = {(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};
            float2 rear = {-(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};

            float2 orig = worlds[wid].robots[rid].transform.pos;
            float angle_robot = atan2(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);

            if (worlds[wid].robots[otherid].front_led == 1)
            {
                float2 dest = transform_mul_vec(worlds[wid].robots[otherid].transform, front);
                float d = distance(orig, dest);

                if (d < CAMERA_RADIUS)
                {
                    float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors |= IN_camera0;
                    }
                    else if ( (angle_dest > angle_robot) &&
                              ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors |= IN_camera1;
                    }
                }
            }

            if (worlds[wid].robots[otherid].rear_led == 1)
            {
                float2 dest = transform_mul_vec(worlds[wid].robots[otherid].transform, rear);
                float d = distance(orig, dest);

                if (d < CAMERA_RADIUS)
                {
                    float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors |= IN_camera2;
                    }
                    else if ( (angle_dest > angle_robot) &&
                         ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors |= IN_camera3;
                    }
                }
            }
        }
    }

    for (i = 0; i < 2; i++)
    {
        float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].target_areas[i].center);

        if (dist < worlds[wid].target_areas[i].radius)
        {
            worlds[wid].robots[rid].sensors |= IN_ground;

            if (worlds[wid].robots[rid].last_target_area < 0)
            {
                worlds[wid].robots[rid].energy = 2;
                worlds[wid].robots[rid].last_target_area = i;
            }
            else if (worlds[wid].robots[rid].last_target_area != i)
            {
                worlds[wid].robots[rid].fitness += worlds[wid].robots[rid].energy;
                worlds[wid].robots[rid].energy = 2;
                worlds[wid].robots[rid].last_target_area = i;
            }
        }

        if (dist < CAMERA_RADIUS + ROBOT_BODY_RADIUS)
        {
            float2 orig = worlds[wid].robots[rid].transform.pos;
            float2 dest = worlds[wid].target_areas[i].center;
            float angle_robot = atan2(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);

            if (distance(orig, dest) < CAMERA_RADIUS)
            {
                float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                if ( (angle_robot > angle_dest) &&
                     ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                {
                    if (!raycast(worlds, orig, dest))
                        worlds[wid].robots[rid].sensors |= IN_camera2;
                }
                else if ( (angle_dest > angle_robot) &&
                     ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                {
                    if (!raycast(worlds, orig, dest))
                        worlds[wid].robots[rid].sensors |= IN_camera3;
                }
            }
        }
    }

    worlds[wid].robots[rid].energy -= (fabs(worlds[wid].robots[rid].wheels_angular_speed.s0) + fabs(worlds[wid].robots[rid].wheels_angular_speed.s1)) /
                                                                        (2 * worlds[wid].k * WHEELS_MAX_ANGULAR_SPEED);
    if (worlds[wid].robots[rid].energy < 0)
        worlds[wid].robots[rid].energy = 0;
}

__kernel void step_robots(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    step_actuators(ranluxcltab, worlds);
    step_dynamics(ranluxcltab, worlds);
    barrier(CLK_GLOBAL_MEM_FENCE);
    step_sensors(ranluxcltab, worlds);
    step_controllers(ranluxcltab, worlds);
}

__kernel void simulate(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int cur = 0;

    while (cur < (TA + TB))
    {
        step_actuators(ranluxcltab, worlds);
        step_dynamics(ranluxcltab, worlds);
        barrier(CLK_GLOBAL_MEM_FENCE);
        step_sensors(ranluxcltab, worlds);
        step_controllers(ranluxcltab, worlds);

        if (cur <= TA) {
            worlds[wid].robots[rid].fitness = 0;
            worlds[wid].robots[rid].energy = 2;
        }

        cur += 1;
    }
}

__kernel void get_transform_matrices(__global world_t *worlds, __global float4 *transforms, __global float *radius)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    transforms[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].transform.pos.x;
    transforms[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].transform.pos.y;
    transforms[wid*ROBOTS_PER_WORLD+rid].s2 = worlds[wid].robots[rid].transform.rot.sin;
    transforms[wid*ROBOTS_PER_WORLD+rid].s3 = worlds[wid].robots[rid].transform.rot.cos;
    radius[wid*ROBOTS_PER_WORLD+rid] = ROBOT_BODY_RADIUS;
}

__kernel void get_individual_fitness_energy(__global world_t *worlds, __global float2 *fitene)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    fitene[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].fitness;
    fitene[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].energy;
}

__kernel void get_world_transforms(__global world_t *worlds, __global float2 *arena, __global float4 *target_areas, __global float2 *target_areas_radius)
{
    int wid = get_global_id(0);

    arena[wid].s0 = worlds[wid].arena_width;
    arena[wid].s1 = worlds[wid].arena_height;

    target_areas[wid].s0 = worlds[wid].target_areas[0].center.x;
    target_areas[wid].s1 = worlds[wid].target_areas[0].center.y;
    target_areas[wid].s2 = worlds[wid].target_areas[1].center.x;
    target_areas[wid].s3 = worlds[wid].target_areas[1].center.y;

    target_areas_radius[wid].s0 = worlds[wid].target_areas[0].radius;
    target_areas_radius[wid].s1 = worlds[wid].target_areas[1].radius;
}

__kernel void get_fitness(__global world_t *worlds, __global float *fitness)
{
    int wid = get_global_id(0);
    int rid;

    float max_trips = ((2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS) * TB * TIME_STEP) / worlds[wid].targets_distance;

    float avg_fitness = 0;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        avg_fitness += worlds[wid].robots[rid].fitness / max_trips;

    fitness[wid] = avg_fitness / ROBOTS_PER_WORLD;
}

__kernel void set_fitness(__global world_t *worlds, float fitness)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    worlds[wid].robots[rid].fitness = fitness;
}

__kernel void set_energy(__global world_t *worlds, float energy)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    worlds[wid].robots[rid].energy = energy;
}

__kernel void get_ann_state(__global world_t *worlds, __global unsigned int *sensors, __global float4 *actuators, __global float4 *hidden)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    sensors[wid*ROBOTS_PER_WORLD+rid] = worlds[wid].robots[rid].sensors;

    actuators[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].actuators[OUT_wheels0];
    actuators[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].actuators[OUT_wheels1];
    actuators[wid*ROBOTS_PER_WORLD+rid].s2 = worlds[wid].robots[rid].actuators[OUT_front_led];
    actuators[wid*ROBOTS_PER_WORLD+rid].s3 = worlds[wid].robots[rid].actuators[OUT_rear_led];

    hidden[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].hidden[HID_hidden0];
    hidden[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].hidden[HID_hidden1];
    hidden[wid*ROBOTS_PER_WORLD+rid].s2 = worlds[wid].robots[rid].hidden[HID_hidden2];
}
