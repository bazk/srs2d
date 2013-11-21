#ifndef __PHYSICS_CL__
#define __PHYSICS_CL__

#pragma OPENCL EXTENSION cl_amd_printf : enable

#ifdef WORK_ITEMS_ARE_WORLDS
    #define WORLD_MEM_SPACE __global
    #define WORLD_ID_GETTER get_global_id
    #define WORLD_BARRIER CLK_GLOBAL_MEM_FENCE
#else
    #define WORLD_MEM_SPACE __local
    #define WORLD_ID_GETTER get_local_id
    #define WORLD_BARRIER CLK_LOCAL_MEM_FENCE
#endif

#include <defs.cl>
#include <util.cl>

#include <pyopencl-ranluxcl.cl>

#include <motor_samples.cl>
#include <ir_wall_samples.cl>
#include <ir_round_samples.cl>

__kernel void init_ranluxcl(uint seed, __global ranluxcl_state_t *ranluxcltab);

__kernel void init_worlds(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float targets_distance);

void init_robot(__global ranluxcl_state_t *ranluxcltab, __global world_t *world, __global robot_t *robot);

void set_random_position(__global ranluxcl_state_t *ranluxcltab, __global world_t *world, __global robot_t *robot);

__kernel void set_ann_parameters(__global world_t *worlds, __global unsigned char *param_list, unsigned int param_size);

float decode_param(unsigned char p, float boundary_l, float boundary_h);

__kernel void get_ann_state(__global world_t *worlds, __global unsigned int *sensors, __global float4 *actuators, __global float4 *hidden);

__kernel void get_transform_matrices(__global world_t *worlds, __global float4 *transforms, __global float *radius);

__kernel void get_world_transforms(__global world_t *worlds, __global float2 *arena, __global float4 *target_areas, __global float2 *target_areas_radius);

__kernel void get_fitness(__global world_t *worlds, __global float *fitness);

__kernel void get_individual_fitness_energy(__global world_t *worlds, __global float2 *fitene);

__kernel void simulate(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds);

__kernel void step_robots(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *worlds, unsigned int currrent_step);

void step_actuators(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);

void step_sensors(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);

void step_collisions(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);

void step_controllers(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot);

bool raycast(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot, float2 p1, float2 p2);

__kernel void init_ranluxcl(uint seed, __global ranluxcl_state_t *ranluxcltab)
{
    ranluxcl_initialization(seed, ranluxcltab);
}

__kernel void init_worlds(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds, float targets_distance)
{
    __global world_t *world = &worlds[get_global_id(0)];

    world->id = get_global_id(0);

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    world->k = (targets_distance / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

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
    world->targets_distance = targets_distance;

    float x = sqrt(pow((targets_distance / 2.0), 2) / 2.0);
    world->target_areas[0].center.x = -x;
    world->target_areas[0].center.y = x;
    world->target_areas[1].center.x = x;
    world->target_areas[1].center.y = -x;

    world->target_areas[0].radius = TARGET_AREAS_RADIUS;
    world->target_areas[1].radius = TARGET_AREAS_RADIUS;

    unsigned int rid;
    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++) {
        world->robots[rid].id = rid;
        init_robot(ranluxcltab, world, &world->robots[rid]);
    }
}

void init_robot(__global ranluxcl_state_t *ranluxcltab, __global world_t *world, __global robot_t *robot)
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

void set_random_position(__global ranluxcl_state_t *ranluxcltab, __global world_t *world, __global robot_t *robot)
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

__kernel void set_ann_parameters(__global world_t *worlds, __global unsigned char *param_list, unsigned int param_size)
{
    __global world_t *world = &worlds[get_global_id(0)];
    __global unsigned char *params = param_list + (get_global_id(0)*param_size);

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
}

float decode_param(unsigned char p, float boundary_l, float boundary_h)
{
    return (float) (p * (boundary_h - boundary_l) / 255) + boundary_l;
}

__kernel void get_ann_state(__global world_t *worlds, __global unsigned int *sensors, __global float4 *actuators, __global float4 *hidden)
{
    __global world_t *world = &worlds[get_global_id(0)];
    __global robot_t *robot;

#ifdef WORK_ITEMS_ARE_WORLDS
    unsigned int rid;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
    {
        robot = &world->robots[rid];
#else
    robot = &world->robots[get_global_id(1)];
#endif

    unsigned int i, s = 0;

    for (i=0; i < NUM_SENSORS; i++)
        if (robot->sensors[i] > 0.5)
            s |= uint_exp2(i);

    sensors[world->id*ROBOTS_PER_WORLD+robot->id] = s;

    actuators[world->id*ROBOTS_PER_WORLD+robot->id].s0 = robot->actuators[OUT_wheels0];
    actuators[world->id*ROBOTS_PER_WORLD+robot->id].s1 = robot->actuators[OUT_wheels1];
    actuators[world->id*ROBOTS_PER_WORLD+robot->id].s2 = robot->actuators[OUT_front_led];
    actuators[world->id*ROBOTS_PER_WORLD+robot->id].s3 = robot->actuators[OUT_rear_led];

    hidden[world->id*ROBOTS_PER_WORLD+robot->id].s0 = robot->hidden[HID_hidden0];
    hidden[world->id*ROBOTS_PER_WORLD+robot->id].s1 = robot->hidden[HID_hidden1];
    hidden[world->id*ROBOTS_PER_WORLD+robot->id].s2 = robot->hidden[HID_hidden2];

#ifdef WORK_ITEMS_ARE_WORLDS
    }
#endif
}

__kernel void get_transform_matrices(__global world_t *worlds, __global float4 *transforms, __global float *radius)
{
    __global world_t *world = &worlds[get_global_id(0)];
    __global robot_t *robot;

#ifdef WORK_ITEMS_ARE_WORLDS
    unsigned int rid;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
    {
        robot = &world->robots[rid];
#else
    robot = &world->robots[get_global_id(1)];
#endif

    transforms[world->id*ROBOTS_PER_WORLD+robot->id].s0 = robot->transform.pos.x;
    transforms[world->id*ROBOTS_PER_WORLD+robot->id].s1 = robot->transform.pos.y;
    transforms[world->id*ROBOTS_PER_WORLD+robot->id].s2 = robot->transform.rot.sin;
    transforms[world->id*ROBOTS_PER_WORLD+robot->id].s3 = robot->transform.rot.cos;
    radius[world->id*ROBOTS_PER_WORLD+robot->id] = ROBOT_BODY_RADIUS;

#ifdef WORK_ITEMS_ARE_WORLDS
    }
#endif
}

__kernel void get_world_transforms(__global world_t *worlds, __global float2 *arena, __global float4 *target_areas, __global float2 *target_areas_radius)
{
    __global world_t *world = &worlds[get_global_id(0)];

    arena[world->id].s0 = world->arena_width;
    arena[world->id].s1 = world->arena_height;

    target_areas[world->id].s0 = world->target_areas[0].center.x;
    target_areas[world->id].s1 = world->target_areas[0].center.y;
    target_areas[world->id].s2 = world->target_areas[1].center.x;
    target_areas[world->id].s3 = world->target_areas[1].center.y;

    target_areas_radius[world->id].s0 = world->target_areas[0].radius;
    target_areas_radius[world->id].s1 = world->target_areas[1].radius;
}

__kernel void get_fitness(__global world_t *worlds, __global float *fitness)
{
    __global world_t *world = &worlds[get_global_id(0)];

    int max_trips = (int) floor( ((2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS) * TB * TIME_STEP) / world->targets_distance );
    float avg_fitness = 0;
    unsigned int rid;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        avg_fitness += world->robots[rid].fitness / max_trips;

    fitness[world->id] = avg_fitness / ROBOTS_PER_WORLD;
}

__kernel void get_individual_fitness_energy(__global world_t *worlds, __global float2 *fitene)
{
    __global world_t *world = &worlds[get_global_id(0)];
    __global robot_t *robot;

#ifdef WORK_ITEMS_ARE_WORLDS
    unsigned int rid;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
    {
        robot = &world->robots[rid];
#else
    robot = &world->robots[get_global_id(1)];
#endif

    fitene[world->id*ROBOTS_PER_WORLD+robot->id].s0 = robot->fitness;
    fitene[world->id*ROBOTS_PER_WORLD+robot->id].s1 = robot->energy;

#ifdef WORK_ITEMS_ARE_WORLDS
    }
#endif
}


__kernel void simulate(__global ranluxcl_state_t *ranluxcltab, __global world_t *worlds)
{
#ifdef WORK_ITEMS_ARE_WORLDS
    unsigned int cur = 0;
    while (cur < (TA + TB))
        step_robots(ranluxcltab, worlds, cur++);

#else // copy world from global to local memory
    __global world_t *world = &worlds[get_global_id(0)];

    __local world_t local_worlds[WORLDS_PER_LOCAL];

    __local world_t *local_world = &local_worlds[get_local_id(0)];

    unsigned int i, j;

    local_world->id = world->id;
    local_world->k = world->k;
    local_world->arena_height = world->arena_height;
    local_world->arena_width = world->arena_width;
    local_world->targets_distance = world->targets_distance;

    for (i=0; i<4; i++)
        local_world->walls[i] = world->walls[i];

    for (i=0; i<2; i++)
        local_world->target_areas[i] = world->target_areas[i];

    for (i=0; i<(NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)); i++)
        local_world->weights[i] = world->weights[i];

    for (i=0; i<NUM_ACTUATORS; i++)
        local_world->bias[i] = world->bias[i];

    for (i=0; i<(NUM_HIDDEN*NUM_SENSORS); i++)
            local_world->weights_hidden[i] = world->weights_hidden[i];

    for (i=0; i<NUM_HIDDEN; i++)
            local_world->bias_hidden[i] = world->bias_hidden[i];

    for (i=0; i<NUM_HIDDEN; i++)
            local_world->timec_hidden[i] = world->timec_hidden[i];


    for (i=0; i<ROBOTS_PER_WORLD; i++)
    {
        __global robot_t *robot = &world->robots[i];
        __local robot_t *local_robot = &local_world->robots[i];

        local_robot->id = robot->id;
        local_robot->transform = robot->transform;
        local_robot->previous_transform = robot->previous_transform;
        local_robot->wheels_angular_speed = robot->wheels_angular_speed;
        local_robot->front_led = robot->front_led;
        local_robot->rear_led = robot->rear_led;
        local_robot->collision = robot->collision;
        local_robot->energy = robot->energy;
        local_robot->fitness = robot->fitness;
        local_robot->last_target_area = robot->last_target_area;
        local_robot->entered_new_target_area = robot->entered_new_target_area;

        for (j=0; j<NUM_SENSORS; j++)
            local_robot->sensors[j] = robot->sensors[j];

        for (j=0; j<NUM_ACTUATORS; j++)
            local_robot->actuators[j] = robot->actuators[j];

        for (j=0; j<NUM_HIDDEN; j++)
            local_robot->hidden[j] = robot->hidden[j];
    }

    unsigned int cur = 0;
    while (cur < (TA + TB))
        step_robots(ranluxcltab, local_worlds, cur++);

    world->id = local_world->id;
    world->k = local_world->k;
    world->arena_height = local_world->arena_height;
    world->arena_width = local_world->arena_width;
    world->targets_distance = local_world->targets_distance;

    for (i=0; i<4; i++)
        world->walls[i] = local_world->walls[i];

    for (i=0; i<2; i++)
        world->target_areas[i] = local_world->target_areas[i];

    for (i=0; i<(NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)); i++)
        world->weights[i] = local_world->weights[i];

    for (i=0; i<NUM_ACTUATORS; i++)
        world->bias[i] = local_world->bias[i];

    for (i=0; i<(NUM_HIDDEN*NUM_SENSORS); i++)
            world->weights_hidden[i] = local_world->weights_hidden[i];

    for (i=0; i<NUM_HIDDEN; i++)
            world->bias_hidden[i] = local_world->bias_hidden[i];

    for (i=0; i<NUM_HIDDEN; i++)
            world->timec_hidden[i] = local_world->timec_hidden[i];


    for (i=0; i<ROBOTS_PER_WORLD; i++)
    {
        __global robot_t *robot = &world->robots[i];
        __local robot_t *local_robot = &local_world->robots[i];

        robot->id = local_robot->id;
        robot->transform = local_robot->transform;
        robot->previous_transform = local_robot->previous_transform;
        robot->wheels_angular_speed = local_robot->wheels_angular_speed;
        robot->front_led = local_robot->front_led;
        robot->rear_led = local_robot->rear_led;
        robot->collision = local_robot->collision;
        robot->energy = local_robot->energy;
        robot->fitness = local_robot->fitness;
        robot->last_target_area = local_robot->last_target_area;
        robot->entered_new_target_area = local_robot->entered_new_target_area;

        for (j=0; j<NUM_SENSORS; j++)
            robot->sensors[j] = local_robot->sensors[j];

        for (j=0; j<NUM_ACTUATORS; j++)
            robot->actuators[j] = local_robot->actuators[j];

        for (j=0; j<NUM_HIDDEN; j++)
            robot->hidden[j] = local_robot->hidden[j];
    }

#endif
}

__kernel void step_robots(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *worlds, unsigned int currrent_step)
{
    WORLD_MEM_SPACE world_t *world = &worlds[WORLD_ID_GETTER(0)];

#ifdef WORK_ITEMS_ARE_WORLDS
    unsigned int rid;

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        step_actuators(ranluxcltab, world, &world->robots[rid]);

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        step_sensors(ranluxcltab, world, &world->robots[rid]);

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        step_collisions(ranluxcltab, world, &world->robots[rid]);

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        step_controllers(ranluxcltab, world, &world->robots[rid]);

    if (currrent_step > TA)
    {
        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        {
            WORLD_MEM_SPACE robot_t *robot = &world->robots[rid];

            robot->energy -= (fabs(robot->wheels_angular_speed.s0) + fabs(robot->wheels_angular_speed.s1)) /
                                                (2 * world->k * WHEELS_MAX_ANGULAR_SPEED);
            if (robot->energy < 0)
                robot->energy = 0;

            if (robot->entered_new_target_area)
            {
                robot->fitness += robot->energy;
                robot->energy = 2;
            }
        }
    }

#else
    WORLD_MEM_SPACE robot_t *robot = &world->robots[WORLD_ID_GETTER(1)];

    step_actuators(ranluxcltab, world, robot);
    barrier(WORLD_BARRIER);
    step_sensors(ranluxcltab, world, robot);
    barrier(WORLD_BARRIER);
    step_collisions(ranluxcltab, world, robot);
    barrier(WORLD_BARRIER);
    step_controllers(ranluxcltab, world, robot);
    barrier(WORLD_BARRIER);

    if (currrent_step > TA)
    {
        robot->energy -= (fabs(robot->wheels_angular_speed.s0) + fabs(robot->wheels_angular_speed.s1)) /
                                            (2 * world->k * WHEELS_MAX_ANGULAR_SPEED);
        if (robot->energy < 0)
            robot->energy = 0;

        if (robot->entered_new_target_area)
        {
            robot->fitness += robot->energy;
            robot->energy = 2;
        }
    }
#endif
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

    float2 pos = { robot->transform.pos.x,
                   robot->transform.pos.y };

    for (i=0; i<NUM_SENSORS; i++)
        robot->sensors[i] = 0;

    robot->entered_new_target_area = 0;

    if ( ((pos.x+(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) > (world->arena_width/2)) ||
         ((pos.x-(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) < (-world->arena_width/2)) ||
         ((pos.y+(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) > (world->arena_height/2)) ||
         ((pos.y-(ROBOT_BODY_RADIUS+IR_WALL_DIST_MAX)) < (-world->arena_height/2)) )
    {
        if ( ((pos.x+ROBOT_BODY_RADIUS) > (world->arena_width/2)) ||
             ((pos.x-ROBOT_BODY_RADIUS) < (-world->arena_width/2)) ||
             ((pos.y+ROBOT_BODY_RADIUS) > (world->arena_height/2)) ||
             ((pos.y-ROBOT_BODY_RADIUS) < (-world->arena_height/2)) )
        {
            robot->collision = 1;
        }

        // IR against 4 walls
        for (i = 0; i < 4; i++)
        {
            float2 proj;

            if ((pos.x >= world->walls[i].p1.x) && (pos.x <= world->walls[i].p2.x))
                proj.x = pos.x;
            else
                proj.x = world->walls[i].p1.x;

            if ((pos.y >= world->walls[i].p1.y) && (pos.y <= world->walls[i].p2.y))
                proj.y = pos.y;
            else
                proj.y = world->walls[i].p1.y;

            float dist = distance(proj, pos) - ROBOT_BODY_RADIUS;

            if (dist <= IR_WALL_DIST_MAX) {
                int dist_idx;

                if (dist <= IR_WALL_DIST_MIN)
                    dist_idx = 0;
                else
                    dist_idx = (int) floor((dist - IR_WALL_DIST_MIN) / IR_WALL_DIST_INTERVAL);

                float wall_angle;
                if (proj.x == pos.x)
                    if (proj.y < pos.y)
                        wall_angle = 3 * M_PI / 2;
                    else
                        wall_angle = M_PI / 2;
                else
                    if (proj.x < pos.x)
                        wall_angle = M_PI;
                    else
                        wall_angle = 0;

                float diff_angle = angle_rot(robot->transform.rot) - wall_angle;

                if (diff_angle >= (2*M_PI))
                    diff_angle -= 2*M_PI;
                else if (diff_angle < 0)
                    diff_angle += 2*M_PI;

                int angle_idx = (int) floor(diff_angle / (2*M_PI / IR_WALL_ANGLE_COUNT));

                for (j = 0; j < 8; j++)
                    robot->sensors[j] += IR_WALL_SAMPLES[dist_idx][angle_idx][j] / 1024.0f;
            }
        }
    }

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        WORLD_MEM_SPACE robot_t *other = &world->robots[otherid];

        if (robot->id == other->id)
            continue;

        float dist = distance(robot->transform.pos, other->transform.pos);

        if (dist < 2*ROBOT_BODY_RADIUS) {
            robot->collision = 1;
        }

        // IR against other robots
        if (dist < (2*ROBOT_BODY_RADIUS+IR_ROUND_DIST_MAX))
        {
            float d = dist - 2*ROBOT_BODY_RADIUS;

            int dist_idx;
            if (d <= IR_ROUND_DIST_MIN)
                dist_idx = 0;
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
        }

        // other robots in camera
        if (dist < (CAMERA_RADIUS + ROBOT_BODY_RADIUS + LED_PROTUBERANCE))
        {
            float2 front = {(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};
            float2 rear = {-(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};

            float2 orig = robot->transform.pos;
            float angle_robot = angle(robot->transform.rot.sin, robot->transform.rot.cos);

            if (other->front_led == 1)
            {
                float2 dest = transform_mul_vec(other->transform, front);
                float d = distance(orig, dest);

                if (d < CAMERA_RADIUS)
                {
                    float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot >= angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(world, robot, orig, dest))
                            robot->sensors[IN_camera0] = 1.0;
                    }
                    if ( (angle_dest >= angle_robot) &&
                              ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(world, robot, orig, dest))
                            robot->sensors[IN_camera1] = 1.0;
                    }
                }
            }

            if (other->rear_led == 1)
            {
                float2 dest = transform_mul_vec(other->transform, rear);
                float d = distance(orig, dest);

                if (d < CAMERA_RADIUS)
                {
                    float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

                    if ( (angle_robot >= angle_dest) &&
                         ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(world, robot, orig, dest))
                            robot->sensors[IN_camera2] = 1.0;
                    }
                    if ( (angle_dest >= angle_robot) &&
                         ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                    {
                        if (!raycast(world, robot, orig, dest))
                            robot->sensors[IN_camera3] = 1.0;
                    }
                }
            }
        }
    }

    for (i = 0; i < 2; i++)
    {
        float dist = distance(robot->transform.pos, world->target_areas[i].center);

        // ground sensor
        if (dist < world->target_areas[i].radius)
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
        if (dist < CAMERA_RADIUS + ROBOT_BODY_RADIUS)
        {
            float2 orig = robot->transform.pos;
            float2 dest = world->target_areas[i].center;
            float angle_robot = angle(robot->transform.rot.sin, robot->transform.rot.cos);

            if (distance(orig, dest) < CAMERA_RADIUS)
            {
                float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

                if ( (angle_robot >= angle_dest) &&
                     ((angle_robot-angle_dest) <= (CAMERA_ANGLE/2)) )
                {
                    if (!raycast(world, robot, orig, dest))
                        robot->sensors[IN_camera2] = 1.0;
                }
                if ( (angle_dest >= angle_robot) &&
                     ((angle_dest-angle_robot) <= (CAMERA_ANGLE/2)) )
                {
                    if (!raycast(world, robot, orig, dest))
                        robot->sensors[IN_camera3] = 1.0;
                }
            }
        }
    }

    for (i=0; i<NUM_SENSORS; i++) {
        if (robot->sensors[i] > 1)
            robot->sensors[i] = 1;
        if (robot->sensors[i] < 0)
            robot->sensors[i] = 0;
    }
}

void step_collisions(__global ranluxcl_state_t *ranluxcltab, WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot)
{
    if (robot->collision != 0) {
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

bool raycast(WORLD_MEM_SPACE world_t *world, WORLD_MEM_SPACE robot_t *robot, float2 p1, float2 p2)
{
    float2 ray = {p2.x - p1.x, p2.y - p1.y};
    float ray_length = length(ray);
    float2 ray_unit = {ray.x / ray_length, ray.y / ray_length};

    unsigned int otherid;
    WORLD_MEM_SPACE robot_t *other;

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        other = &world->robots[otherid];

        if (robot->id == other->id)
            continue;

        float dist = distance(p1, other->transform.pos);

        if (dist < ray_length + ROBOT_BODY_RADIUS)
        {
            float2 v1 = { other->transform.pos.x - p1.x,
                          other->transform.pos.y - p1.y };

            float proj = v1.x * ray_unit.x + v1.y * ray_unit.y;

            if ((proj > 0) && (proj < ray_length)) {
                float2 d = {ray_unit.x * proj, ray_unit.y * proj};
                d.x += p1.x;
                d.y += p1.y;

                if (distance(d, other->transform.pos) < ROBOT_BODY_RADIUS)
                    return true;
            }
        }
    }

    return false;
}

#endif
