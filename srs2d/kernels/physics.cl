#ifndef __PHYSICS_CL__
#define __PHYSICS_CL__

#pragma OPENCL EXTENSION cl_amd_printf : enable

#if defined(WORK_ITEMS_ARE_WORLDS) || defined(NO_LOCAL)
    #define WORLD_MEM_SPACE __global
#else
    #define WORLD_MEM_SPACE __local
#endif

#include <defs.cl>
#include <util.cl>

#include <ranluxcl.cl>

#include <motor_samples.cl>
#include <ir_wall_samples.cl>
#include <ir_round_samples.cl>

void init_world(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, float targets_distance, __global unsigned char *params);
void init_robot(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
void set_random_position(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
void step_actuators(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
void step_sensors(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
void step_collisions(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
void step_controllers(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
// bool raycast(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot, float2 p1, float2 p2);
void fill_dist_table(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);
void fill_raycast_table(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);

__kernel void init_ranluxcl(uint seed, __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_initialization(seed, ranluxcltab);
}

__kernel void simulate(__global ranluxcl_state_t *ranluxcltab,
                       __global world_t *worlds,
                       float targets_distance,
                       __global unsigned char *param_list,
                       unsigned int param_size,

                       // return variables
                       __global float *fitness,
                       __global float *robot_radius,
                       __global float2 *arena_size,
                       __global float2 *target_areas_pos,
                       __global float *target_areas_radius,
                       __global float *fitness_hist,
                       __global float *energy_hist,
                       __global float4 *transform_hist,
                       __global float *sensors_hist,
                       __global float *actuators_hist,
                       __global float *hidden_hist,
                       unsigned int save_hist
                       )
{
    __global unsigned char *params = param_list + (get_global_id(0)*param_size);

    unsigned int cur = 0;
    unsigned int rid;

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    float k = (targets_distance / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

    // max_trips = maximum number of trips a robot, at maximum speed, can perform during a simulation of TB time steps
    int max_trips = (int) floor( ((2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS) * TB * TIME_STEP) / targets_distance );

#if defined(WORK_ITEMS_ARE_WORLDS) || defined(NO_LOCAL)
    __global world_t *world = &worlds[get_global_id(0)];

    world->id = get_global_id(0);
    init_world(ranluxcltab, world, targets_distance, params);

    while (cur < (TA + TB))
    {
        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_actuators(ranluxcltab, world, &world->robots[rid]);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_sensors(ranluxcltab, world, &world->robots[rid]);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_collisions(ranluxcltab, world, &world->robots[rid]);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_controllers(ranluxcltab, world, &world->robots[rid]);

        if (cur > TA)
        {
            for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            {
                WORLD_MEM_SPACE robot_t *robot = &world->robots[rid];

                robot->energy -= (fabs(robot->wheels_angular_speed.s0) + fabs(robot->wheels_angular_speed.s1)) /
                                                    (2 * k * WHEELS_MAX_ANGULAR_SPEED);
                if (robot->energy < 0)
                    robot->energy = 0;

                if (robot->entered_new_target_area)
                {
                    robot->fitness += robot->energy;
                    robot->energy = 2;
                }
            }
        }

        if (save_hist == 1)
        {
            unsigned int idx = cur * (NUM_WORLDS * ROBOTS_PER_WORLD) + world->id * ROBOTS_PER_WORLD;
            unsigned int idx2;
            unsigned int i;

            robot_radius[world->id] = ROBOT_BODY_RADIUS;
            arena_size[world->id].x = world->arena_width;
            arena_size[world->id].y = world->arena_height;
            target_areas_pos[world->id*2] = world->target_areas[0].center;
            target_areas_pos[world->id*2+1] = world->target_areas[1].center;
            target_areas_radius[world->id*2] = world->target_areas[0].radius;
            target_areas_radius[world->id*2+1] = world->target_areas[1].radius;

            for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            {
                fitness_hist[idx+rid] = world->robots[rid].fitness;
                energy_hist[idx+rid] = world->robots[rid].energy;
                transform_hist[idx+rid].s0 = world->robots[rid].transform.pos.x;
                transform_hist[idx+rid].s1 = world->robots[rid].transform.pos.y;
                transform_hist[idx+rid].s2 = world->robots[rid].transform.rot.sin;
                transform_hist[idx+rid].s3 = world->robots[rid].transform.rot.cos;

                idx2 = cur * (NUM_WORLDS * ROBOTS_PER_WORLD * NUM_SENSORS) + world->id * (ROBOTS_PER_WORLD * NUM_SENSORS) + rid * NUM_SENSORS;
                for (i=0; i<NUM_SENSORS; i++)
                    sensors_hist[idx2+i] = world->robots[rid].sensors[i];

                idx2 = cur * (NUM_WORLDS * ROBOTS_PER_WORLD * NUM_ACTUATORS) + world->id * (ROBOTS_PER_WORLD * NUM_ACTUATORS) + rid * NUM_ACTUATORS;
                for (i=0; i<NUM_ACTUATORS; i++)
                    actuators_hist[idx2+i] = world->robots[rid].actuators[i];

                idx2 = cur * (NUM_WORLDS * ROBOTS_PER_WORLD * NUM_HIDDEN) + world->id * (ROBOTS_PER_WORLD * NUM_HIDDEN) + rid * NUM_HIDDEN;
                for (i=0; i<NUM_HIDDEN; i++)
                    hidden_hist[idx2+i] = world->robots[rid].hidden[i];
            }
        }

        cur++;
    }

    float avg_fitness = 0;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        avg_fitness += world->robots[rid].fitness / max_trips;

    fitness[world->id] = avg_fitness / ROBOTS_PER_WORLD;

#else
    __local world_t local_worlds[WORLDS_PER_LOCAL];

    __local world_t *world = &local_worlds[get_local_id(0)];
    __local robot_t *robot = &world->robots[get_local_id(1)];

    if (get_local_id(1) == 0)
    {
        world->id = get_global_id(0);
        init_world(ranluxcltab, world, targets_distance, params);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    while (cur < (TA + TB))
    {
        step_actuators(ranluxcltab, world, robot);
        barrier(CLK_LOCAL_MEM_FENCE);

        step_sensors(ranluxcltab, world, robot);
        barrier(CLK_LOCAL_MEM_FENCE);

        step_collisions(ranluxcltab, world, robot);
        barrier(CLK_LOCAL_MEM_FENCE);

        step_controllers(ranluxcltab, world, robot);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (cur > TA)
        {
            robot->energy -= (fabs(robot->wheels_angular_speed.s0) + fabs(robot->wheels_angular_speed.s1)) /
                                                (2 * k * WHEELS_MAX_ANGULAR_SPEED);
            if (robot->energy < 0)
                robot->energy = 0;

            if (robot->entered_new_target_area)
            {
                robot->fitness += robot->energy;
                robot->energy = 2;
            }
        }

        if (save_hist == 1)
        {
            unsigned int idx = cur * (NUM_WORLDS * ROBOTS_PER_WORLD) + world->id * ROBOTS_PER_WORLD;
            unsigned int idx2;
            unsigned int i;

            if (robot->id == 0)
            {
                robot_radius[world->id] = ROBOT_BODY_RADIUS;
                arena_size[world->id].x = world->arena_width;
                arena_size[world->id].y = world->arena_height;
                target_areas_pos[world->id*2] = world->target_areas[0].center;
                target_areas_pos[world->id*2+1] = world->target_areas[1].center;
                target_areas_radius[world->id*2] = world->target_areas[0].radius;
                target_areas_radius[world->id*2+1] = world->target_areas[1].radius;
            }

            fitness_hist[idx+robot->id] = robot->fitness;
            energy_hist[idx+robot->id] = robot->energy;
            transform_hist[idx+robot->id].s0 = robot->transform.pos.x;
            transform_hist[idx+robot->id].s1 = robot->transform.pos.y;
            transform_hist[idx+robot->id].s2 = robot->transform.rot.sin;
            transform_hist[idx+robot->id].s3 = robot->transform.rot.cos;

            idx2 = cur * (NUM_WORLDS * ROBOTS_PER_WORLD * NUM_SENSORS) + world->id * (ROBOTS_PER_WORLD * NUM_SENSORS) + robot->id * NUM_SENSORS;
            for (i=0; i<NUM_SENSORS; i++)
                sensors_hist[idx2+i] = robot->sensors[i];

            idx2 = cur * (NUM_WORLDS * ROBOTS_PER_WORLD * NUM_ACTUATORS) + world->id * (ROBOTS_PER_WORLD * NUM_ACTUATORS) + robot->id * NUM_ACTUATORS;
            for (i=0; i<NUM_ACTUATORS; i++)
                actuators_hist[idx2+i] = robot->actuators[i];

            idx2 = cur * (NUM_WORLDS * ROBOTS_PER_WORLD * NUM_HIDDEN) + world->id * (ROBOTS_PER_WORLD * NUM_HIDDEN) + robot->id * NUM_HIDDEN;
            for (i=0; i<NUM_HIDDEN; i++)
                hidden_hist[idx2+i] = robot->hidden[i];
        }

        cur++;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(1) == 0)
    {
        float avg_fitness = 0;

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            avg_fitness += world->robots[rid].fitness / max_trips;

        fitness[world->id] = avg_fitness / ROBOTS_PER_WORLD;
    }
#endif
}

void init_world(__global ranluxcl_state_t *ranluxcltab,
                WORLD_MEM_SPACE world_t *world,
                float targets_distance,
                __global unsigned char *params)
{
    // walls
    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
    float4 random = ranluxcl32(&ranluxclstate);
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);

    world->arena_height = ARENA_HEIGHT;
    world->arena_width = ARENA_WIDTH_MIN + random.s0 * (ARENA_WIDTH_MAX - ARENA_WIDTH_MIN);

    world->walls[0].p1.x = world->arena_width / 2;
    world->walls[0].p1.y = -world->arena_height / 2;
    world->walls[0].p2.x = world->arena_width / 2;
    world->walls[0].p2.y = world->arena_height / 2;

    world->walls[1].p1.x = -world->arena_width / 2;
    world->walls[1].p1.y = world->arena_height / 2;
    world->walls[1].p2.x = world->arena_width / 2;
    world->walls[1].p2.y = world->arena_height / 2;

    world->walls[2].p1.x = -world->arena_width / 2;
    world->walls[2].p1.y = -world->arena_height / 2;
    world->walls[2].p2.x = -world->arena_width / 2;
    world->walls[2].p2.y = world->arena_height / 2;

    world->walls[3].p1.x = -world->arena_width / 2;
    world->walls[3].p1.y = -world->arena_height / 2;
    world->walls[3].p2.x = world->arena_width / 2;
    world->walls[3].p2.y = -world->arena_height / 2;

    // target areas
    // float x = sqrt(pow((targets_distance / 2.0), 2) / 2.0);
    world->target_areas[0].center.x = 0;
    world->target_areas[0].center.y = targets_distance / 2;
    world->target_areas[1].center.x = 0;
    world->target_areas[1].center.y = -targets_distance / 2;

    world->target_areas[0].radius = TARGET_AREAS_RADIUS;
    world->target_areas[1].radius = TARGET_AREAS_RADIUS;

    unsigned int i, j, p = 0;

    for (i=0; i<NUM_ACTUATORS; i++)
    {
        for (j=0; j<(NUM_SENSORS+NUM_HIDDEN); j++)
            world->weights[i*(NUM_SENSORS+NUM_HIDDEN)+j] = decode_param(params[p++], WEIGHTS_BOUNDARY_L, WEIGHTS_BOUNDARY_H);

        world->bias[i] = decode_param(params[p++], BIAS_BOUNDARY_L, BIAS_BOUNDARY_H);
    }

    for (i=0; i<NUM_HIDDEN; i++)
    {
        for (j=0; j<NUM_SENSORS; j++)
            world->weights_hidden[i*NUM_SENSORS+j] = decode_param(params[p++], WEIGHTS_BOUNDARY_L, WEIGHTS_BOUNDARY_H);

        world->bias_hidden[i] = decode_param(params[p++], BIAS_BOUNDARY_L, BIAS_BOUNDARY_H);
        world->timec_hidden[i] = decode_param(params[p++], TIMEC_BOUNDARY_L, TIMEC_BOUNDARY_H);
    }

    unsigned int rid;
    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
    {
        world->robots[rid].id = rid;
        init_robot(ranluxcltab, world, &world->robots[rid]);
    }
}

void init_robot(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    unsigned int i;

    robot->wheels_angular_speed.s0 = 0;
    robot->wheels_angular_speed.s1 = 0;
    robot->front_led = 0;
    robot->rear_led = 0;
    robot->collision = 0;
    robot->energy = 2;
    robot->fitness = 0;
    robot->last_target_area = -1;
    robot->entered_new_target_area = 0;
    set_random_position(ranluxcltab, world, robot);

    for (i=0; i<NUM_SENSORS; i++)
        robot->sensors[i] = 0;

    for (i=0; i<NUM_ACTUATORS; i++)
        robot->actuators[i] = 0;

    for (i=0; i<NUM_HIDDEN; i++)
        robot->hidden[i] = 0;

#ifdef TEST
    if (robot->id == 0) {
        robot->transform.pos.x = 0;
        robot->transform.pos.y = 0;
        robot->transform.rot.sin = 0;
        robot->transform.rot.cos = 1;
    }
    else if (robot->id == 1) {
        robot->transform.pos.x = 0.073;
        robot->transform.pos.y = 0;
        robot->transform.rot.sin = 1;
        robot->transform.rot.cos = 0;
    }

    robot->previous_transform.pos.x = robot->transform.pos.x;
    robot->previous_transform.pos.y = robot->transform.pos.y;
    robot->previous_transform.rot.sin = robot->transform.rot.sin;
    robot->previous_transform.rot.cos = robot->transform.rot.cos;
#endif
}

void set_random_position(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    int otherid, i, collision = 1;

    ranluxcl_state_t ranluxclstate;
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

    while (collision == 1)
    {
        float4 random = ranluxcl32(&ranluxclstate);

        float max_x = (world->arena_width / 2) - ROBOT_BODY_RADIUS;
        float max_y = (world->arena_height / 2) - ROBOT_BODY_RADIUS;
        robot->transform.pos.x = (random.s0 * 2 * max_x) - max_x;
        robot->transform.pos.y = (random.s1 * 2 * max_y) - max_y;
        robot->transform.rot.sin = random.s2 * 2 - 1;
        robot->transform.rot.cos = random.s3 * 2 - 1;

        robot->previous_transform.pos.x = robot->transform.pos.x;
        robot->previous_transform.pos.y = robot->transform.pos.y;
        robot->previous_transform.rot.sin = robot->transform.rot.sin;
        robot->previous_transform.rot.cos = robot->transform.rot.cos;

        collision = 0;

        // check for collision with other robots
        for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
        {
            if (robot->id != world->robots[otherid].id)
            {
                float dist = distance(robot->transform.pos, world->robots[otherid].transform.pos);

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
            float dist = distance(robot->transform.pos, world->target_areas[i].center);

            if (dist < world->target_areas[i].radius)
            {
                collision = 1;
                break;
            }
        }
    }

    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
}

void step_actuators(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    robot->previous_transform.pos.x = robot->transform.pos.x;
    robot->previous_transform.pos.y = robot->transform.pos.y;
    robot->previous_transform.rot.sin = robot->transform.rot.sin;
    robot->previous_transform.rot.cos = robot->transform.rot.cos;

    robot->front_led = (robot->actuators[OUT_front_led] > 0.5) ? 1 : 0;
    robot->rear_led = (robot->actuators[OUT_rear_led] > 0.5) ? 1 : 0;

    robot->wheels_angular_speed.s0 = (robot->actuators[OUT_wheels0] * 2 * WHEELS_MAX_ANGULAR_SPEED) - WHEELS_MAX_ANGULAR_SPEED;
    robot->wheels_angular_speed.s1 = (robot->actuators[OUT_wheels1] * 2 * WHEELS_MAX_ANGULAR_SPEED) - WHEELS_MAX_ANGULAR_SPEED;

    int v1 = round(robot->actuators[OUT_wheels1] * (MOTOR_SAMPLE_COUNT - 1));
    int v2 = round(robot->actuators[OUT_wheels0] * (MOTOR_SAMPLE_COUNT - 1));

    robot->transform.pos.x += MOTOR_LINEAR_SPEED_SAMPLES[v1][v2] * robot->transform.rot.cos * TIME_STEP;
    robot->transform.pos.y += MOTOR_LINEAR_SPEED_SAMPLES[v1][v2] * robot->transform.rot.sin * TIME_STEP;

    float angle_robot = angle(robot->transform.rot.sin, robot->transform.rot.cos);
    angle_robot += MOTOR_ANGULAR_SPEED_SAMPLES[v1][v2] * TIME_STEP;
    robot->transform.rot.sin = sin(angle_robot);
    robot->transform.rot.cos = cos(angle_robot);
}

void step_sensors(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    unsigned int i, j, otherid;
    WORLD_MEM_SPACE robot_t *other;

    // float2 pos = { robot->transform.pos.x,
    //                robot->transform.pos.y };

    for (i=0; i<NUM_SENSORS; i++)
        robot->sensors[i] = 0;

    robot->entered_new_target_area = 0;

    fill_dist_table(world, robot);
    fill_raycast_table(world, robot);

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        other = &world->robots[otherid];

        if (robot->id == other->id)
            continue;

        if (robot->dist_table[otherid] < (2*ROBOT_BODY_RADIUS))
            robot->collision = 1;

        // IR against other robots
        float d = robot->dist_table[otherid] - 2*ROBOT_BODY_RADIUS;

        int dist_idx;
        if (d <= IR_ROUND_DIST_MIN)
            dist_idx = 0;
        else if (d >= IR_ROUND_DIST_MAX)
            dist_idx = IR_ROUND_DIST_COUNT-1;
        else
            dist_idx = (int) floor((d - IR_ROUND_DIST_MIN) / IR_ROUND_DIST_INTERVAL);

        float s = other->transform.pos.y - robot->transform.pos.y;
        float c = other->transform.pos.x - robot->transform.pos.x;
        float diff_angle = angle_rot(robot->transform.rot) - angle(s, c);

        if (diff_angle >= (2*M_PI))
            diff_angle -= 2*M_PI;
        else if (diff_angle < 0)
            diff_angle += 2*M_PI;

        int angle_idx = (int) floor(diff_angle / ((2*M_PI) / IR_ROUND_ANGLE_COUNT));

        for (j = 0; j < 8; j++)
            robot->sensors[j] += IR_ROUND_SAMPLES[dist_idx][angle_idx][j] / 1024.0f;

        // camera
        robot->sensors[IN_camera0] += ((robot->raycast_table[otherid] & 1) != 0) ? 1 : 0;
        robot->sensors[IN_camera1] += ((robot->raycast_table[otherid] & 2) != 0) ? 1 : 0;
        robot->sensors[IN_camera2] += ((robot->raycast_table[otherid] & 4) != 0) ? 1 : 0;
        robot->sensors[IN_camera3] += ((robot->raycast_table[otherid] & 8) != 0) ? 1 : 0;

    // if ( ((pos.x+(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) > (world->arena_width/2)) ||
    //      ((pos.x-(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) < (-world->arena_width/2)) ||
    //      ((pos.y+(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) > (world->arena_height/2)) ||
    //      ((pos.y-(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) < (-world->arena_height/2)) )
    // {
    //     if ( ((pos.x+ROBOT_BODY_RADIUS) > (world->arena_width/2)) ||
    //          ((pos.x-ROBOT_BODY_RADIUS) < (-world->arena_width/2)) ||
    //          ((pos.y+ROBOT_BODY_RADIUS) > (world->arena_height/2)) ||
    //          ((pos.y-ROBOT_BODY_RADIUS) < (-world->arena_height/2)) )
    //     {
    //         robot->collision = 1;
    //     }

    //     // IR against 4 walls
    //     for (i = 0; i < 4; i++)
    //     {
    //         float2 proj;

    //         if ((pos.x >= world->walls[i].p1.x) && (pos.x <= world->walls[i].p2.x))
    //             proj.x = pos.x;
    //         else
    //             proj.x = world->walls[i].p1.x;

    //         if ((pos.y >= world->walls[i].p1.y) && (pos.y <= world->walls[i].p2.y))
    //             proj.y = pos.y;
    //         else
    //             proj.y = world->walls[i].p1.y;

    //         float dist = distance(proj, pos) - ROBOT_BODY_RADIUS;

    //         if (dist <= IR_WALL_DIST_MAX) {
    //             int dist_idx;

    //             if (dist <= IR_WALL_DIST_MIN)
    //                 dist_idx = 0;
    //             else
    //                 dist_idx = (int) floor((dist - IR_WALL_DIST_MIN) / IR_WALL_DIST_INTERVAL);

    //             float wall_angle;
    //             if (proj.x == pos.x)
    //                 if (proj.y < pos.y)
    //                     wall_angle = 3 * M_PI / 2;
    //                 else
    //                     wall_angle = M_PI / 2;
    //             else
    //                 if (proj.x < pos.x)
    //                     wall_angle = M_PI;
    //                 else
    //                     wall_angle = 0;

    //             float diff_angle = angle_rot(robot->transform.rot) - wall_angle;

    //             if (diff_angle >= (2*M_PI))
    //                 diff_angle -= 2*M_PI;
    //             else if (diff_angle < 0)
    //                 diff_angle += 2*M_PI;

    //             int angle_idx = (int) floor(diff_angle / (2*M_PI / IR_WALL_ANGLE_COUNT));

    //             for (j = 0; j < 8; j++)
    //                 robot->sensors[j] += IR_WALL_SAMPLES[dist_idx][angle_idx][j] / 1024.0f;
    //         }
    //     }
    // }
    }

    for (i = 0; i < 2; i++)
    {
        // ground sensor
        if (robot->dist_table[ROBOTS_PER_WORLD+i] < world->target_areas[i].radius)
        {
            robot->sensors[IN_ground] = 1.0;

            if (robot->last_target_area != i)
            {
                if (robot->last_target_area >= 0)
                    robot->entered_new_target_area = 1;

                robot->last_target_area = i;
            }
        }

        // target area led in camera
        robot->sensors[IN_camera2] += ((robot->raycast_table[ROBOTS_PER_WORLD+i] & 4) != 0) ? 1 : 0;
        robot->sensors[IN_camera3] += ((robot->raycast_table[ROBOTS_PER_WORLD+i] & 8) != 0) ? 1 : 0;
    }

    for (i=0; i<NUM_SENSORS; i++)
    {
        if (robot->sensors[i] > 1)
            robot->sensors[i] = 1;
        if (robot->sensors[i] < 0)
            robot->sensors[i] = 0;
    }
}

void step_collisions(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    if (robot->collision != 0)
    {
        robot->collision = 0;

        robot->transform.pos.x = robot->previous_transform.pos.x;
        robot->transform.pos.y = robot->previous_transform.pos.y;
        robot->transform.rot.sin = robot->previous_transform.rot.sin;
        robot->transform.rot.cos = robot->previous_transform.rot.cos;

        robot->wheels_angular_speed.s0 = 0;
        robot->wheels_angular_speed.s1 = 0;
    }
}

void step_controllers(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    unsigned int s,h,a;
    float aux;

    for (h=0; h<NUM_HIDDEN; h++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += world->weights_hidden[h*NUM_SENSORS+s] * robot->sensors[s];

        aux += world->bias_hidden[h];

        robot->hidden[h] = (world->timec_hidden[h] * robot->hidden[h]) + ((1 - world->timec_hidden[h]) * sigmoid(aux));
    }

    for (a=0; a<NUM_ACTUATORS; a++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += world->weights[a*(NUM_SENSORS+NUM_HIDDEN)+s] * robot->sensors[s];

        for (h=0; h<NUM_HIDDEN; h++)
            aux += world->weights[a*(NUM_SENSORS+NUM_HIDDEN)+NUM_SENSORS+h] * robot->hidden[h];

        aux += world->bias[a];

        robot->actuators[a] = sigmoid(aux);
    }

#ifdef TEST
    if (robot->id == 0) {
        robot->actuators[OUT_wheels0] = 0.4;
        robot->actuators[OUT_wheels1] = 0.6;
    }
    else if (robot->id == 1) {
        robot->actuators[OUT_wheels0] = 0.72;
        robot->actuators[OUT_wheels1] = 0.9;
    }
    else {
        robot->actuators[OUT_wheels0] = 1;
        robot->actuators[OUT_wheels1] = 1;
    }
#endif
}

// bool raycast(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot, float2 p1, float2 p2)
// {
//     float2 ray = {p2.x - p1.x, p2.y - p1.y};
//     float ray_length = length(ray);
//     float2 ray_unit = {ray.x / ray_length, ray.y / ray_length};

//     unsigned int otherid;
//     WORLD_MEM_SPACE robot_t *other;

//     for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
//     {
//         other = &world->robots[otherid];

//         if (robot->id == other->id)
//             continue;

//         float dist = distance(p1, other->transform.pos);

//         if (dist < ray_length + ROBOT_BODY_RADIUS)
//         {
//             float2 v1 = { other->transform.pos.x - p1.x,
//                           other->transform.pos.y - p1.y };

//             float proj = v1.x * ray_unit.x + v1.y * ray_unit.y;

//             if ((proj > 0) && (proj < ray_length)) {
//                 float2 d = {ray_unit.x * proj, ray_unit.y * proj};
//                 d.x += p1.x;
//                 d.y += p1.y;

//                 if (distance(d, other->transform.pos) < ROBOT_BODY_RADIUS)
//                     return true;
//             }
//         }
//     }

//     return false;
// }

void fill_dist_table(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    unsigned int otherid, targetid;

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
        robot->dist_table[otherid] = distance(robot->transform.pos, world->robots[otherid].transform.pos);

    for (targetid = 0; targetid < 2; targetid++)
        robot->dist_table[ROBOTS_PER_WORLD+targetid] = distance(robot->transform.pos, world->target_areas[targetid].center);
}

void fill_raycast_table(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    unsigned int otherid, interid, targetid;
    float2 otherpos, interpos, targetpos;

    float2 front = {ROBOT_BODY_RADIUS, 0};
    float2 rear = {-ROBOT_BODY_RADIUS, 0};

    float2 camerapos = robot->transform.pos + front;

    float robot_angle = angle_rot(robot->transform.rot);

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        if (robot->id == otherid)
            continue;

        otherpos = world->robots[otherid].transform.pos;

        float2 front_ray = (otherpos + front) - camerapos;
        float front_ray_length = length(front_ray);
        float2 front_ray_normal = front_ray / front_ray_length;

        float2 rear_ray = (otherpos + rear) - camerapos;
        float rear_ray_length = length(rear_ray);
        float2 rear_ray_normal = rear_ray / rear_ray_length;

        unsigned int front_intercept = 0, rear_intercept = 0;

        for (interid = 0; interid < ROBOTS_PER_WORLD; interid++)
        {
            if (otherid == interid)
                continue;

            interpos = world->robots[interid].transform.pos;

            float2 v1 = interpos - camerapos;

            float front_t = dot(v1, front_ray_normal);
            float2 front_proj = (front_ray_normal * front_t) + camerapos;

            float rear_t = dot(v1, rear_ray_normal);
            float2 rear_proj = (rear_ray_normal * rear_t) + camerapos;

            front_intercept += ((front_t > 0) && (front_t < front_ray_length) &&
                               (distance(front_proj, interpos) < ROBOT_BODY_RADIUS)) ? 1 : 0;

            rear_intercept += ((rear_t > 0) && (rear_t < rear_ray_length) &&
                              (distance(rear_proj, interpos) < ROBOT_BODY_RADIUS)) ? 1 : 0;
        }

        // if (robot->id == 0)
        //     printf("[%d] %d %d\n", otherid, front_intercept, rear_intercept);

        float front_angle = angle(front_ray_normal.y, front_ray_normal.x) - robot_angle;
        float rear_angle = angle(rear_ray_normal.y, rear_ray_normal.x) - robot_angle;

        robot->raycast_table[otherid] = 0;

        if ((fabs(front_angle) < (CAMERA_ANGLE/2)) && (front_ray_length < CAMERA_RADIUS) && (front_intercept == 0))
        {
            if (front_angle < 0)
                robot->raycast_table[otherid] |= 1;
            else
                robot->raycast_table[otherid] |= 2;
        }

        if ((fabs(rear_angle) < (CAMERA_ANGLE/2)) && (rear_ray_length < CAMERA_RADIUS) && (rear_intercept == 0))
        {
            if (rear_angle < 0)
                robot->raycast_table[otherid] |= 4;
            else
                robot->raycast_table[otherid] |= 8;
        }
    }

    for (targetid = 0; targetid < 2; targetid++)
    {
        targetpos = world->target_areas[targetid].center;

        float2 ray = targetpos - camerapos;
        float ray_length = length(ray);
        float2 ray_normal = ray / ray_length;

        unsigned int intercept = 0;

        for (interid = 0; interid < ROBOTS_PER_WORLD; interid++)
        {
            interpos = world->robots[interid].transform.pos;

            float2 v1 = interpos - camerapos;

            float t = dot(v1, ray_normal);
            float2 proj = (ray_normal * t) + camerapos;

            intercept += ((t > 0) && (t < ray_length) &&
                          (distance(proj, interpos) < ROBOT_BODY_RADIUS)) ? 1 : 0;
        }

        float target_angle = angle(ray_normal.y, ray_normal.x) - robot_angle;

        robot->raycast_table[ROBOTS_PER_WORLD+targetid] = 0;

        if ((fabs(target_angle) < (CAMERA_ANGLE/2)) && (ray_length < CAMERA_RADIUS) && (intercept == 0))
        {
            if (target_angle < 0)
                robot->raycast_table[ROBOTS_PER_WORLD+targetid] |= 4;
            else
                robot->raycast_table[ROBOTS_PER_WORLD+targetid] |= 8;
        }
    }
}

#endif
