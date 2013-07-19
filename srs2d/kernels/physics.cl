#include <pyopencl-ranluxcl.cl>

#define ROBOT_BODY_RADIUS           0.06
#define WHEELS_MAX_ANGULAR_SPEED    12.565
#define WHEELS_DISTANCE             0.0825
#define WHEELS_RADIUS               0.02
#define IR_RADIUS                   0.025 // from center of the robot
#define CAMERA_RADIUS               0.35 // from center of the robot
#define CAMERA_ANGLE                1.2566370614359172 // 72 degrees
#define LED_PROTUBERANCE            0.01 // from outside border of the robot
#define TARGET_AREAS_RADIUS         0.27

#define NUM_SENSORS    13
#define NUM_ACTUATORS   4
#define NUM_HIDDEN      3

#define IN_camera0      0
#define IN_camera1      1
#define IN_camera2      2
#define IN_camera3      3
#define IN_proximity0   4
#define IN_proximity1   5
#define IN_proximity2   6
#define IN_proximity3   7
#define IN_proximity4   8
#define IN_proximity5   9
#define IN_proximity6  10
#define IN_proximity7  11
#define IN_ground      12

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
    int collision_count;
    float energy;
    float fitness;
    int last_target_area;

    float sensors[NUM_SENSORS];
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

    target_area_t target_areas[2];

    // ANN parameters
    float weights[NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)];
    float bias[NUM_ACTUATORS];

    float weights_hidden[NUM_HIDDEN*NUM_SENSORS];
    float bias_hidden[NUM_HIDDEN];
    float timec_hidden[NUM_HIDDEN];
} world_t;

// TODO: obstacles

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

float raycast(__global robot_t *robots, float2 p1, float2 p2)
{
    int world = get_global_id(0);
    int robot = get_global_id(1);
    int id = world * ROBOTS_PER_WORLD + robot;

    unsigned int i, other;
    float ray_dist = distance(p1, p2);
    float min = ray_dist;
    float2 d = {p2.x - p1.x, p2.y - p1.y}; // vector of ray, from start to end
    float a = (d.x * d.x) + (d.y * d.y);   // d dot d

    for (i = 0; i < ROBOTS_PER_WORLD; i++)
    {
        other = world * ROBOTS_PER_WORLD + i;

        if (id == other)
            continue;

        float dist = distance(p1, robots[other].transform.pos);

        if (dist < ray_dist + ROBOT_BODY_RADIUS, 2)
        {
            float2 f = {p1.x - robots[other].transform.pos.x,
                        p1.y - robots[other].transform.pos.y }; // vector from center sphere to ray start

            float b = 2 * ((f.x * d.x) + (f.y * d.y)); // f dot d
            float c = ((f.x * f.x) + (f.y * f.y)) - ROBOT_BODY_RADIUS*ROBOT_BODY_RADIUS; // f dot f - r**2

            float discriminant = (b*b - 4*a*c);

            if (discriminant < 0)
                continue;

            discriminant = sqrt(discriminant);
            float x1 = (-b - discriminant) / (2*a);
            float x2 = (-b + discriminant) / (2*a);

            if ((x1 >= 0 && x1 <= 1) || (x2 >= 0 && x2 <= 1))
                if (dist < min)
                    min = dist;
        }
    }

    return min;
}

__kernel void init_ranluxcl(uint seed, __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_initialization(seed, ranluxcltab);
}

__kernel void init_worlds(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float targets_distance)
{
    int wid = get_global_id(0);

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    worlds[wid].k = (targets_distance / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

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

    // random_position
    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
    float4 random = ranluxcl32(&ranluxclstate);
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

    worlds[wid].robots[rid].transform.pos.x = random.s0 * 2 - 2;
    worlds[wid].robots[rid].transform.pos.y = random.s1 * 2 - 2;
    worlds[wid].robots[rid].transform.rot.sin = random.s2 * 2 - 1;
    worlds[wid].robots[rid].transform.rot.cos = random.s3 * 2 - 1;
    worlds[wid].robots[rid].wheels_angular_speed.s0 = 0;
    worlds[wid].robots[rid].wheels_angular_speed.s1 = 0;
    worlds[wid].robots[rid].front_led = 0;
    worlds[wid].robots[rid].rear_led = 0;
    worlds[wid].robots[rid].collision_count = 0;
    worlds[wid].robots[rid].energy = 2;
    worlds[wid].robots[rid].fitness = 0.0;
    worlds[wid].robots[rid].last_target_area = -1;

    for (i=0; i<NUM_SENSORS; i++)
        worlds[wid].robots[rid].sensors[i] = 0;

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
        for (j=0; j<NUM_SENSORS+NUM_HIDDEN; j++)
            worlds[wid].weights[i*NUM_ACTUATORS+j] = weights[i*NUM_ACTUATORS+j];

        worlds[wid].bias[i] = weights[i];
    }

    for (i=0; i<NUM_HIDDEN; i++)
    {
        for (j=0; j<NUM_SENSORS; j++)
            worlds[wid].weights_hidden[i*NUM_HIDDEN+j] = weights_hidden[i*NUM_HIDDEN+j];

        worlds[wid].bias_hidden[i] = bias_hidden[i];
        worlds[wid].timec_hidden[i] = timec_hidden[i];
    }
}

void step_controllers(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int s,h,a;
    float aux;

    for (h=0; h<NUM_HIDDEN; h++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += worlds[wid].weights_hidden[h*NUM_HIDDEN+s] *
                  worlds[wid].robots[rid].sensors[s] + worlds[wid].bias_hidden[h];

        worlds[wid].robots[rid].hidden[h] = worlds[wid].timec_hidden[h] * worlds[wid].robots[rid].hidden[h] +
              (1 - worlds[wid].timec_hidden[h]) * sigmoid(aux);
    }

    for (a=0; a<NUM_ACTUATORS; a++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += worlds[wid].weights[a*NUM_ACTUATORS+s] * worlds[wid].robots[rid].sensors[s];

        for (h=0; h<NUM_HIDDEN; h++)
            aux += worlds[wid].weights[a*NUM_ACTUATORS+h+NUM_SENSORS] * worlds[wid].robots[rid].hidden[h];

        aux += worlds[wid].bias[a];

        worlds[wid].robots[rid].actuators[a] = sigmoid(aux);
    }
}

void step_actuators(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    worlds[wid].robots[rid].wheels_angular_speed.s0 = worlds[wid].robots[rid].actuators[OUT_wheels0] * WHEELS_MAX_ANGULAR_SPEED;
    worlds[wid].robots[rid].wheels_angular_speed.s1 = worlds[wid].robots[rid].actuators[OUT_wheels1] * WHEELS_MAX_ANGULAR_SPEED;
    worlds[wid].robots[rid].front_led = worlds[wid].robots[rid].actuators[OUT_front_led];
    worlds[wid].robots[rid].rear_led = worlds[wid].robots[rid].actuators[OUT_rear_led];
}

void step_dynamics(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float time_step)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    float2 wls; // wheels linear speed
    wls.s0 = worlds[wid].robots[rid].wheels_angular_speed.s0 * WHEELS_RADIUS;
    wls.s1 = worlds[wid].robots[rid].wheels_angular_speed.s1 * WHEELS_RADIUS;

    float2 wlv = {wls.s0 + wls.s1, 0}; // wheels linear velocity

    float2 linear_velocity = rot_mul_vec(worlds[wid].robots[rid].transform.rot, wlv);

    float angular_speed = atan2(wls.s0 - wls.s1, (float) WHEELS_DISTANCE / 2);

    worlds[wid].robots[rid].transform.pos.x += linear_velocity.x * time_step;
    worlds[wid].robots[rid].transform.pos.y += linear_velocity.y * time_step;

    float angle = atan2(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);
    angle += angular_speed * time_step;
    worlds[wid].robots[rid].transform.rot.sin = sin(angle);
    worlds[wid].robots[rid].transform.rot.cos = cos(angle);
}

void step_sensors(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int i, otherid;

    for (i = 0; i < NUM_SENSORS; i++)
        worlds[wid].robots[rid].sensors[i] = 0.0;

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        if (rid == otherid)
            continue;

        float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].robots[otherid].transform.pos);

        if (otherid > rid)
        {
            if (dist < 2*ROBOT_BODY_RADIUS)
            {
                // random_position
                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
                float4 random = ranluxcl32(&ranluxclstate);
                ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

                worlds[wid].robots[rid].transform.pos.x = random.s0 * 4 - 2;
                worlds[wid].robots[rid].transform.pos.y = random.s1 * 4 - 2;
                worlds[wid].robots[rid].transform.rot.sin = random.s2 * 2 - 1;
                worlds[wid].robots[rid].transform.rot.cos = random.s3 * 2 - 1;
            }
        }

        if (dist < 2*ROBOT_BODY_RADIUS+IR_RADIUS)
        {
            float s = worlds[wid].robots[otherid].transform.pos.y - worlds[wid].robots[rid].transform.pos.y;
            float c = worlds[wid].robots[otherid].transform.pos.x - worlds[wid].robots[rid].transform.pos.x;
            float a = atan2(s, c) - atan2(worlds[wid].robots[otherid].transform.rot.sin, worlds[wid].robots[otherid].transform.rot.cos);
            int idx = (int) floor(a / (2*M_PI/8)) % 8;
            worlds[wid].robots[rid].sensors[IN_proximity0+idx] = 1 - ((dist - 2*ROBOT_BODY_RADIUS) / IR_RADIUS);
        }

        if (dist < CAMERA_RADIUS + ROBOT_BODY_RADIUS + LED_PROTUBERANCE)
        {
            float2 front = {(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};
            float2 rear = {-(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};

            float2 orig = worlds[wid].robots[rid].transform.pos;
            float angle_robot = atan2(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);

            if (worlds[wid].robots[otherid].actuators[OUT_front_led] > 0.5)
            {
                float2 dest = transform_mul_vec(worlds[wid].robots[otherid].transform, front);

                if (distance(orig, dest) < CAMERA_RADIUS)
                {
                    float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        worlds[wid].robots[rid].sensors[IN_camera0] = 1 - (raycast(worlds[wid].robots, orig, dest) / CAMERA_RADIUS);
                    }
                    else if ( (angle_dest > angle_robot) &&
                              ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        worlds[wid].robots[rid].sensors[IN_camera1] = 1 - (raycast(worlds[wid].robots, orig, dest) / CAMERA_RADIUS);
                    }
                }
            }

            if (worlds[wid].robots[otherid].actuators[OUT_rear_led] > 0.5)
            {
                float2 dest = transform_mul_vec(worlds[wid].robots[otherid].transform, rear);

                if (distance(orig, dest) < CAMERA_RADIUS)
                {
                    float angle_dest = atan2(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        worlds[wid].robots[rid].sensors[IN_camera2] = 1 - (raycast(worlds[wid].robots, orig, dest) / CAMERA_RADIUS);
                    }
                    else if ( (angle_dest > angle_robot) &&
                         ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        worlds[wid].robots[rid].sensors[IN_camera3] = 1 - (raycast(worlds[wid].robots, orig, dest) / CAMERA_RADIUS);
                    }
                }
            }
        }
    }

    for (i = 0; i < 2; i++)
    {
        float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].target_areas[i].center);

        if (dist < pow(worlds[wid].target_areas[i].radius, 2))
        {
            worlds[wid].robots[rid].sensors[IN_ground] = 1;

            if (worlds[wid].robots[rid].last_target_area != i)
            {
                worlds[wid].robots[rid].fitness += worlds[wid].robots[rid].energy;
                worlds[wid].robots[rid].energy = 2;
                worlds[wid].robots[rid].last_target_area = i;
            }
            else
                worlds[wid].robots[rid].energy -= (fabs(worlds[wid].robots[rid].wheels_angular_speed.s0) + fabs(worlds[wid].robots[rid].wheels_angular_speed.s1) /
                                                                                (2 * worlds[wid].k * WHEELS_MAX_ANGULAR_SPEED));
        }
        else
        {
            worlds[wid].robots[rid].sensors[IN_ground] = 0;
            worlds[wid].robots[rid].energy -= (fabs(worlds[wid].robots[rid].wheels_angular_speed.s0) + fabs(worlds[wid].robots[rid].wheels_angular_speed.s1) /
                                                                            (2 * worlds[wid].k * WHEELS_MAX_ANGULAR_SPEED));
        }
    }
}

__kernel void step_robots(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    unsigned int i;

    step_actuators(ranluxcltab, worlds);

    for (i=0; i < DYNAMICS_ITERATIONS; i++)
        step_dynamics(ranluxcltab, worlds, TIME_STEP/DYNAMICS_ITERATIONS);

    step_sensors(ranluxcltab, worlds);

    step_controllers(ranluxcltab, worlds);
}

__kernel void simulate(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float seconds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int i, cur = 0, max_steps = ceil(seconds / TIME_STEP);

    while (cur < max_steps)
    {
        if (cur == floor(max_steps / 2.0))
            worlds[wid].robots[rid].fitness = 0;

        step_actuators(ranluxcltab, worlds);

        for (i=0; i < DYNAMICS_ITERATIONS; i++)
            step_dynamics(ranluxcltab, worlds, TIME_STEP/DYNAMICS_ITERATIONS);

        step_sensors(ranluxcltab, worlds);

        step_controllers(ranluxcltab, worlds);

        cur += 1;
    }
}

__kernel void get_transform_matrices(__global world_t *worlds, __global float4 *transforms)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    transforms[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].transform.pos.x;
    transforms[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].transform.pos.y;
    transforms[wid*ROBOTS_PER_WORLD+rid].s2 = worlds[wid].robots[rid].transform.rot.sin;
    transforms[wid*ROBOTS_PER_WORLD+rid].s3 = worlds[wid].robots[rid].transform.rot.cos;
}


__kernel void get_fitness(__global world_t *worlds, __global float *fitness)
{
    int wid = get_global_id(0);
    int rid;

    float avg_fitness = 0;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        avg_fitness += worlds[wid].robots[rid].fitness;

    fitness[wid] = avg_fitness / ROBOTS_PER_WORLD;
}