#ifndef __PHYSICS_CL__
#define __PHYSICS_CL__

#pragma OPENCL EXTENSION cl_amd_printf : enable

#include <defs.cl>
#include <util.cl>

#include <pyopencl-ranluxcl.cl>

#include <motor_samples.cl>
#include <ir_wall_samples.cl>
#include <ir_round_samples.cl>

#include <raycast.cl>

__kernel void size_of_world_t(__global int *result)
{
    *result = (int) sizeof(world_t);
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

__kernel void init_arenas(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);

    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
    float4 random = ranluxcl32(&ranluxclstate);
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

    worlds[wid].arena_height = ARENA_HEIGHT;
    worlds[wid].arena_width = ARENA_WIDTH_MIN + random.s0 * (ARENA_WIDTH_MAX - ARENA_WIDTH_MIN);

    worlds[wid].walls[0].p1.x = worlds[wid].arena_width / 2;
    worlds[wid].walls[0].p1.y = -worlds[wid].arena_height / 2;
    worlds[wid].walls[0].p2.x = worlds[wid].arena_width / 2;
    worlds[wid].walls[0].p2.y = worlds[wid].arena_height / 2;
    worlds[wid].walls[0].rot.sin = 1;
    worlds[wid].walls[0].rot.cos = 0;

    worlds[wid].walls[1].p1.x = -worlds[wid].arena_width / 2;
    worlds[wid].walls[1].p1.y = worlds[wid].arena_height / 2;
    worlds[wid].walls[1].p2.x = worlds[wid].arena_width / 2;
    worlds[wid].walls[1].p2.y = worlds[wid].arena_height / 2;
    worlds[wid].walls[1].rot.sin = 0;
    worlds[wid].walls[1].rot.cos = -1;

    worlds[wid].walls[2].p1.x = -worlds[wid].arena_width / 2;
    worlds[wid].walls[2].p1.y = -worlds[wid].arena_height / 2;
    worlds[wid].walls[2].p2.x = -worlds[wid].arena_width / 2;
    worlds[wid].walls[2].p2.y = worlds[wid].arena_height / 2;
    worlds[wid].walls[2].rot.sin = -1;
    worlds[wid].walls[2].rot.cos = 0;

    worlds[wid].walls[3].p1.x = -worlds[wid].arena_width / 2;
    worlds[wid].walls[3].p1.y = -worlds[wid].arena_height / 2;
    worlds[wid].walls[3].p2.x = worlds[wid].arena_width / 2;
    worlds[wid].walls[3].p2.y = -worlds[wid].arena_height / 2;
    worlds[wid].walls[3].rot.sin = 0;
    worlds[wid].walls[3].rot.cos = 1;
}

__kernel void init_worlds(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float targets_distance)
{
    int wid = get_global_id(0);

    worlds[wid].targets_distance = targets_distance;

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    worlds[wid].k = (targets_distance / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

    init_arenas(ranluxcltab, worlds);

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
    worlds[wid].robots[rid].collision_count = 0;

    for (i=0; i<NUM_SENSORS; i++)
        worlds[wid].robots[rid].sensors[i] = 0;

    for (i=0; i<NUM_ACTUATORS; i++)
        worlds[wid].robots[rid].actuators[i] = 0;

    for (i=0; i<NUM_HIDDEN; i++)
        worlds[wid].robots[rid].hidden[i] = 0;

#ifdef TEST
    if (rid == 0) {
        worlds[wid].robots[rid].transform.pos.x = 0;
        worlds[wid].robots[rid].transform.pos.y = 0;
        worlds[wid].robots[rid].transform.rot.sin = 0;
        worlds[wid].robots[rid].transform.rot.cos = 1;
    }
    else if (rid == 1) {
        worlds[wid].robots[rid].transform.pos.x = 0.15;
        worlds[wid].robots[rid].transform.pos.y = 0;
        worlds[wid].robots[rid].transform.rot.sin = 1;
        worlds[wid].robots[rid].transform.rot.cos = 0;
    }
#endif
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
            aux += worlds[wid].weights_hidden[h*NUM_SENSORS+s] * worlds[wid].robots[rid].sensors[s];

        aux += worlds[wid].bias_hidden[h];

        worlds[wid].robots[rid].hidden[h] = worlds[wid].timec_hidden[h] * worlds[wid].robots[rid].hidden[h] +
              (1 - worlds[wid].timec_hidden[h]) * sigmoid(aux);
    }

    for (a=0; a<NUM_ACTUATORS; a++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += worlds[wid].weights[a*(NUM_SENSORS+NUM_HIDDEN)+s] * worlds[wid].robots[rid].sensors[s];

        for (h=0; h<NUM_HIDDEN; h++)
            aux += worlds[wid].weights[a*(NUM_SENSORS+NUM_HIDDEN)+NUM_SENSORS+h] * worlds[wid].robots[rid].hidden[h];

        aux += worlds[wid].bias[a];

        worlds[wid].robots[rid].actuators[a] = sigmoid(aux);
    }

#ifdef TEST
    if (rid == 0) {
        worlds[wid].robots[rid].actuators[OUT_wheels0] = 0.5;
        worlds[wid].robots[rid].actuators[OUT_wheels1] = 0.5;
    }
    else {
        worlds[wid].robots[rid].actuators[OUT_wheels0] = 0.8;
        worlds[wid].robots[rid].actuators[OUT_wheels1] = 0.9;
    }
#endif
}

__kernel void step_actuators(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    worlds[wid].robots[rid].wheels_angular_speed.s0 = (worlds[wid].robots[rid].actuators[OUT_wheels0] * 2 * WHEELS_MAX_ANGULAR_SPEED) - WHEELS_MAX_ANGULAR_SPEED;
    worlds[wid].robots[rid].wheels_angular_speed.s1 = (worlds[wid].robots[rid].actuators[OUT_wheels1] * 2 * WHEELS_MAX_ANGULAR_SPEED) - WHEELS_MAX_ANGULAR_SPEED;
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
        worlds[wid].robots[rid].collision_count += 1;
        return;
    }

    float dt = 1.0f; // should be TIME_STEP (needs interpolation)

    int v1 = round(worlds[wid].robots[rid].actuators[OUT_wheels1] * (MOTOR_SAMPLE_COUNT - 1));
    int v2 = round(worlds[wid].robots[rid].actuators[OUT_wheels0] * (MOTOR_SAMPLE_COUNT - 1));

    float angular_speed = MOTOR_ANGULAR_SPEED_SAMPLES[v1][v2] * M_PI / 180.0f;
    float linear_speed = MOTOR_LINEAR_SPEED_SAMPLES[v1][v2] / 100.0f;

    worlds[wid].robots[rid].transform.pos.x += linear_speed * worlds[wid].robots[rid].transform.rot.cos * dt;
    worlds[wid].robots[rid].transform.pos.y += linear_speed * worlds[wid].robots[rid].transform.rot.sin * dt;

    float angle_robot = angle(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);
    angle_robot += angular_speed * dt;
    worlds[wid].robots[rid].transform.rot.sin = sin(angle_robot);
    worlds[wid].robots[rid].transform.rot.cos = cos(angle_robot);
}

__kernel void step_sensors(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
    int wid = get_global_id(0);
    int rid = get_global_id(1);

    unsigned int i, j, otherid;

    float2 pos = { worlds[wid].robots[rid].transform.pos.x,
                   worlds[wid].robots[rid].transform.pos.y };

    for (i=0; i<NUM_SENSORS; i++)
        worlds[wid].robots[rid].sensors[i] = 0;

    if ( ((pos.x+(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) > (worlds[wid].arena_width/2)) ||
         ((pos.x-(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) < (-worlds[wid].arena_width/2)) ||
         ((pos.y+(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) > (worlds[wid].arena_height/2)) ||
         ((pos.y-(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) < (-worlds[wid].arena_height/2)) )
    {
        if ( ((pos.x+ROBOT_BODY_RADIUS) > (worlds[wid].arena_width/2)) ||
             ((pos.x-ROBOT_BODY_RADIUS) < (-worlds[wid].arena_width/2)) ||
             ((pos.y+ROBOT_BODY_RADIUS) > (worlds[wid].arena_height/2)) ||
             ((pos.y-ROBOT_BODY_RADIUS) < (-worlds[wid].arena_height/2)) )
        {
            worlds[wid].robots[rid].collision = 1;
            return;
        }

        // IR against 4 walls
        for (i = 0; i < 4; i++)
        {
            float2 proj;

            if ((pos.x >= worlds[wid].walls[i].p1.x) && (pos.x <= worlds[wid].walls[i].p2.x))
                proj.x = pos.x;
            else
                proj.x = worlds[wid].walls[i].p1.x;

            if ((pos.y >= worlds[wid].walls[i].p1.y) && (pos.y <= worlds[wid].walls[i].p2.y))
                proj.y = pos.y;
            else
                proj.y = worlds[wid].walls[i].p1.y;

            float dist = distance(proj, pos) - ROBOT_BODY_RADIUS;

            if (dist <= IR_WALL_DIST_MAX) {
                if (dist < IR_WALL_DIST_MIN)
                    dist = IR_WALL_DIST_MIN;

                int dist_idx = (int) floor((dist - IR_WALL_DIST_MIN) / IR_WALL_DIST_INTERVAL);

                float diff_angle = angle_rot(worlds[wid].robots[rid].transform.rot) - angle_rot(worlds[wid].walls[i].rot);

                if (diff_angle >= (2*M_PI))
                    diff_angle -= 2*M_PI;
                else if (diff_angle < 0)
                    diff_angle += 2*M_PI;

                int angle_idx = (int) floor(diff_angle / (2*M_PI / IR_WALL_ANGLE_COUNT));

                for (j = 0; j < 8; j++)
                    worlds[wid].robots[rid].sensors[j] += IR_WALL_SAMPLES[dist_idx][angle_idx][j] / 1024.0f;
            }
        }
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

        // IR against other robots
        if (dist < (2*ROBOT_BODY_RADIUS+IR_ROUND_DIST_MAX))
        {
            float d = dist - 2*ROBOT_BODY_RADIUS;
            if (d < IR_ROUND_DIST_MIN)
                d = IR_ROUND_DIST_MIN;

            int dist_idx = (int) floor((d - IR_ROUND_DIST_MIN) / IR_ROUND_DIST_INTERVAL);

            float s = worlds[wid].robots[otherid].transform.pos.y - worlds[wid].robots[rid].transform.pos.y;
            float c = worlds[wid].robots[otherid].transform.pos.x - worlds[wid].robots[rid].transform.pos.x;
            float diff_angle = angle_rot(worlds[wid].robots[rid].transform.rot) - angle(s, c) - (M_PI / 2.0f);

            if (diff_angle >= (2*M_PI))
                diff_angle -= 2*M_PI;
            else if (diff_angle < 0)
                diff_angle += 2*M_PI;

            int angle_idx = (int) floor(diff_angle / (2*M_PI / IR_ROUND_ANGLE_COUNT));

            for (j = 0; j < 8; j++)
                worlds[wid].robots[rid].sensors[j] += IR_ROUND_SAMPLES[dist_idx][angle_idx][j] / 1024.0f;
        }

        // other robots in camera
        if (dist < CAMERA_RADIUS + ROBOT_BODY_RADIUS + LED_PROTUBERANCE)
        {
            float2 front = {(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};
            float2 rear = {-(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};

            float2 orig = worlds[wid].robots[rid].transform.pos;
            float angle_robot = angle(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);

            if (worlds[wid].robots[otherid].front_led == 1)
            {
                float2 dest = transform_mul_vec(worlds[wid].robots[otherid].transform, front);
                float d = distance(orig, dest);

                if (d < CAMERA_RADIUS)
                {
                    float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors[IN_camera0] = 1.0;
                    }
                    else if ( (angle_dest >= angle_robot) &&
                              ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors[IN_camera1] = 1.0;
                    }
                }
            }

            if (worlds[wid].robots[otherid].rear_led == 1)
            {
                float2 dest = transform_mul_vec(worlds[wid].robots[otherid].transform, rear);
                float d = distance(orig, dest);

                if (d < CAMERA_RADIUS)
                {
                    float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot > angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors[IN_camera2] = 1.0;
                    }
                    else if ( (angle_dest >= angle_robot) &&
                         ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(worlds, orig, dest))
                            worlds[wid].robots[rid].sensors[IN_camera3] = 1.0;
                    }
                }
            }
        }
    }

    for (i = 0; i < 2; i++)
    {
        float dist = distance(worlds[wid].robots[rid].transform.pos, worlds[wid].target_areas[i].center);

        // ground sensor
        if (dist < worlds[wid].target_areas[i].radius)
        {
            worlds[wid].robots[rid].sensors[IN_ground] = 1.0;

            if (worlds[wid].robots[rid].last_target_area < 0)
            {
                worlds[wid].robots[rid].energy = 2;
                worlds[wid].robots[rid].last_target_area = i;
            }
            else if (worlds[wid].robots[rid].last_target_area != i)
            {
                worlds[wid].robots[rid].fitness += worlds[wid].robots[rid].energy;
                worlds[wid].robots[rid].energy = 2;
                // worlds[wid].robots[rid].fitness += 1;
                worlds[wid].robots[rid].last_target_area = i;
            }
        }


        // target area led in camera
        if (dist < CAMERA_RADIUS + ROBOT_BODY_RADIUS)
        {
            float2 orig = worlds[wid].robots[rid].transform.pos;
            float2 dest = worlds[wid].target_areas[i].center;
            float angle_robot = angle(worlds[wid].robots[rid].transform.rot.sin, worlds[wid].robots[rid].transform.rot.cos);

            if (distance(orig, dest) < CAMERA_RADIUS)
            {
                float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

                if ( (angle_robot > angle_dest) &&
                     ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                {
                    if (!raycast(worlds, orig, dest))
                        worlds[wid].robots[rid].sensors[IN_camera2] = 1.0;
                }
                else if ( (angle_dest >= angle_robot) &&
                     ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                {
                    if (!raycast(worlds, orig, dest))
                        worlds[wid].robots[rid].sensors[IN_camera3] = 1.0;
                }
            }
        }
    }

    for (i=0; i<NUM_SENSORS; i++) {
        if (worlds[wid].robots[rid].sensors[i] > 1)
            worlds[wid].robots[rid].sensors[i] = 1;
        if (worlds[wid].robots[rid].sensors[i] < 0)
            worlds[wid].robots[rid].sensors[i] = 0;
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

    int max_trips = floor( ((2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS) * TB * TIME_STEP) / worlds[wid].targets_distance );

    float avg_fitness = 0;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++) {
        // if (worlds[wid].robots[rid].collision_count > 0) {
        //     // if one robot collided, fitness is zeroed
        //     fitness[wid] = 0;
        //     return;
        // }

        // avg_fitness += worlds[wid].robots[rid].fitness / max_trips;
        if (worlds[wid].robots[rid].collision_count <= 0)
            avg_fitness += 1;
    }

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

    unsigned int i, s = 0;

    for (i=0; i < NUM_SENSORS; i++)
        if (worlds[wid].robots[rid].sensors[i] > 0.5)
            s |= uint_exp2(i);

    sensors[wid*ROBOTS_PER_WORLD+rid] = s;

    actuators[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].actuators[OUT_wheels0];
    actuators[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].actuators[OUT_wheels1];
    actuators[wid*ROBOTS_PER_WORLD+rid].s2 = worlds[wid].robots[rid].actuators[OUT_front_led];
    actuators[wid*ROBOTS_PER_WORLD+rid].s3 = worlds[wid].robots[rid].actuators[OUT_rear_led];

    hidden[wid*ROBOTS_PER_WORLD+rid].s0 = worlds[wid].robots[rid].hidden[HID_hidden0];
    hidden[wid*ROBOTS_PER_WORLD+rid].s1 = worlds[wid].robots[rid].hidden[HID_hidden1];
    hidden[wid*ROBOTS_PER_WORLD+rid].s2 = worlds[wid].robots[rid].hidden[HID_hidden2];
}

#endif