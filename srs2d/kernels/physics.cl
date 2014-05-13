#ifndef __PHYSICS_CL__
#define __PHYSICS_CL__

#pragma OPENCL EXTENSION cl_amd_printf : enable

#include <defs.cl>
#include <util.cl>

#include <motor_samples.cl>
#include <ir_wall_samples.cl>
#include <ir_round_samples.cl>

void init_world(__global float *random, __global world_t *world, __local transform_t *transforms, float targets_distance, float targets_angle, __global float *params);
void init_robot(__global float *random, __global world_t *world, __local transform_t *transforms, __global robot_t *robot);
void set_random_position(__global float *random, __global world_t *world, __local transform_t *transforms, __global robot_t *robot);
void step_actuators(__global world_t *world, __local transform_t *transforms, __global robot_t *robot);
void step_sensors(__global world_t *world, __local transform_t *transforms, __global robot_t *robot);
void step_collisions(__global world_t *world, __local transform_t *transforms, __global robot_t *robot);
void step_controllers(__global world_t *world, __local transform_t *transforms, __global robot_t *robot);
void fill_raycast_table(__global world_t *world, __local transform_t *transforms, __global robot_t *robot);

__kernel
__attribute__((reqd_work_group_size(WORLDS_PER_LOCAL, ROBOTS_PER_LOCAL, 1)))
void simulate(__global float *random,
              __global world_t *worlds,
              float targets_distance,
              float targets_angle,
              __global float *param_list,
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
    __global float *params = &param_list[get_global_id(0)*param_size];

    unsigned int cur = 0;
    unsigned int rid;

    __global world_t *world = &worlds[get_global_id(0)];
    __global robot_t *robot = &world->robots[get_global_id(1)];

    __local transform_t local_transforms[WORLDS_PER_LOCAL][ROBOTS_PER_WORLD];
    __local transform_t *transforms = local_transforms[get_local_id(0)];

#ifdef WORK_ITEMS_ARE_WORLDS
    world->id = get_global_id(0);
    init_world(random, world, transforms, targets_distance, targets_angle, params);

    for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
    {
        world->robots[rid].id = rid;
        init_robot(random, world, transforms, &world->robots[rid]);
    }

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    float k = (distance(world->target_areas[0].center, world->target_areas[1].center) / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

    // max_trips = maximum number of trips a robot, at maximum speed, can perform during a simulation of TB time steps
    int max_trips = (int) floor( ((2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS) * TB * TIME_STEP) / distance(world->target_areas[0].center, world->target_areas[1].center) );

    while (cur < (TA + TB))
    {
        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_actuators(world, transforms, &world->robots[rid]);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_sensors(world, transforms, &world->robots[rid]);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_collisions(world, transforms, &world->robots[rid]);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            step_controllers(world, transforms, &world->robots[rid]);

        step_world(world);

        if (cur > TA)
        {
            for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            {
                robot = &world->robots[rid];

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

        if ((world->id == 0) && (save_hist == 1))
        {
            unsigned int idx = cur*ROBOTS_PER_WORLD;
            unsigned int idx2;
            unsigned int i;

            robot_radius[0] = ROBOT_BODY_RADIUS;
            arena_size[0].x = world->arena_width;
            arena_size[0].y = world->arena_height;
            target_areas_pos[0] = world->target_areas[0].center;
            target_areas_pos[1] = world->target_areas[1].center;
            target_areas_radius[0] = world->target_areas[0].radius;
            target_areas_radius[1] = world->target_areas[1].radius;

            for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            {
                fitness_hist[idx+rid] = world->robots[rid].fitness;
                energy_hist[idx+rid] = world->robots[rid].energy;
                transform_hist[idx+rid].s0 = transforms[rid].pos.x;
                transform_hist[idx+rid].s1 = transforms[rid].pos.y;
                transform_hist[idx+rid].s2 = transforms[rid].rot.sin;
                transform_hist[idx+rid].s3 = transforms[rid].rot.cos;

                idx2 = cur * ROBOTS_PER_WORLD * NUM_SENSORS + rid * NUM_SENSORS;
                for (i=0; i<NUM_SENSORS; i++)
                    sensors_hist[idx2+i] = world->robots[rid].sensors[i];

                idx2 = cur * ROBOTS_PER_WORLD * NUM_ACTUATORS + rid * NUM_ACTUATORS;
                for (i=0; i<NUM_ACTUATORS; i++)
                    actuators_hist[idx2+i] = world->robots[rid].actuators[i];

                idx2 = cur * ROBOTS_PER_WORLD * NUM_HIDDEN + rid * NUM_HIDDEN;
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

    if (get_global_id(1) == 0)
    {
        world->id = get_global_id(0);
        init_world(random, world, transforms, targets_distance, targets_angle, params);

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
        {
            world->robots[rid].id = rid;
            init_robot(random, world, transforms, &world->robots[rid]);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // k = number of time steps needed for a robot to consume one unit of energy while moving at maximum speed
    float k = (distance(world->target_areas[0].center, world->target_areas[1].center) / (2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS)) / TIME_STEP;

    // max_trips = maximum number of trips a robot, at maximum speed, can perform during a simulation of TB time steps
    int max_trips = (int) floor( ((2 * WHEELS_MAX_ANGULAR_SPEED * WHEELS_RADIUS) * TB * TIME_STEP) / distance(world->target_areas[0].center, world->target_areas[1].center) );

    while (cur < (TA + TB))
    {
        step_actuators(world, transforms, robot);
        barrier(CLK_GLOBAL_MEM_FENCE);

        step_sensors(world, transforms, robot);
        barrier(CLK_GLOBAL_MEM_FENCE);

        step_collisions(world, transforms, robot);
        barrier(CLK_GLOBAL_MEM_FENCE);

        step_controllers(world, transforms, robot);
        barrier(CLK_GLOBAL_MEM_FENCE);

        step_world(world);
        barrier(CLK_GLOBAL_MEM_FENCE);

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

        if ((world->id == 0) && (save_hist == 1))
        {
            unsigned int idx = cur*ROBOTS_PER_WORLD;
            unsigned int idx2;
            unsigned int i;

            if (robot->id == 0)
            {
                robot_radius[0] = ROBOT_BODY_RADIUS;
                arena_size[0].x = world->arena_width;
                arena_size[0].y = world->arena_height;
                target_areas_pos[0] = world->target_areas[0].center;
                target_areas_pos[1] = world->target_areas[1].center;
                target_areas_radius[0] = world->target_areas[0].radius;
                target_areas_radius[1] = world->target_areas[1].radius;
            }

            fitness_hist[idx+robot->id] = robot->fitness;
            energy_hist[idx+robot->id] = robot->energy;
            transform_hist[idx+robot->id].s0 = transforms[robot->id].pos.x;
            transform_hist[idx+robot->id].s1 = transforms[robot->id].pos.y;
            transform_hist[idx+robot->id].s2 = transforms[robot->id].rot.sin;
            transform_hist[idx+robot->id].s3 = transforms[robot->id].rot.cos;

            idx2 = cur * ROBOTS_PER_WORLD * NUM_SENSORS + robot->id * NUM_SENSORS;
            for (i=0; i<NUM_SENSORS; i++)
                sensors_hist[idx2+i] = robot->sensors[i];

            idx2 = cur * ROBOTS_PER_WORLD * NUM_ACTUATORS + robot->id * NUM_ACTUATORS;
            for (i=0; i<NUM_ACTUATORS; i++)
                actuators_hist[idx2+i] = robot->actuators[i];

            idx2 = cur * ROBOTS_PER_WORLD * NUM_HIDDEN + robot->id * NUM_HIDDEN;
            for (i=0; i<NUM_HIDDEN; i++)
                hidden_hist[idx2+i] = robot->hidden[i];
        }

        cur++;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (get_global_id(1) == 0)
    {
        float avg_fitness = 0;

        for (rid = 0; rid < ROBOTS_PER_WORLD; rid++)
            avg_fitness += world->robots[rid].fitness / max_trips;

        fitness[world->id] = avg_fitness / ROBOTS_PER_WORLD;
    }
#endif
}

void init_world(__global float *random,
                __global world_t *world,
                __local transform_t *transforms,
                float targets_distance,
                float targets_angle,
                __global float *params)
{
    world->random_offset = 0;

    // walls
    world->arena_height = ARENA_HEIGHT;
    world->arena_width = ARENA_WIDTH_MIN + random[world->id*NUM_WORLDS+(world->random_offset++)] * (ARENA_WIDTH_MAX - ARENA_WIDTH_MIN);

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

    world->target_areas[0].radius = TARGET_AREAS_RADIUS;
    world->target_areas[1].radius = TARGET_AREAS_RADIUS;

    world->targets_distance = targets_distance;
    world->targets_angle = targets_angle;

#if defined(RANDOM_TARGET_AREAS) && defined(SYMETRICAL_TARGET_AREAS)
    world->targets_distance = (random[world->id*NUM_WORLDS+(world->random_offset++)] * 0.6) + 0.8;
    world->targets_angle = (random[world->id*NUM_WORLDS+(world->random_offset++)] * M_PI);
#endif

    world->target_areas[0].center.x = cos(world->targets_angle) * (world->targets_distance / 2);
    world->target_areas[0].center.y = sin(world->targets_angle) * (world->targets_distance / 2);
    world->target_areas[1].center.x = -cos(world->targets_angle) * (world->targets_distance / 2);
    world->target_areas[1].center.y = -sin(world->targets_angle) * (world->targets_distance / 2);

#if defined(RANDOM_TARGET_AREAS) && (!defined(SYMETRICAL_TARGET_AREAS))
    float max_x = (world->arena_width / 2) - TARGET_AREAS_RADIUS;
    float max_y = (world->arena_height / 2) - TARGET_AREAS_RADIUS;
    world->target_areas[0].center.x = (random[world->id*NUM_WORLDS+(world->random_offset++)] * 2 * max_x) - max_x;
    world->target_areas[0].center.y = (random[world->id*NUM_WORLDS+(world->random_offset++)] * 2 * max_y) - max_y;

    float gap_width = world->targets_distance + TARGET_AREAS_RADIUS - (world->arena_width / 2);
    float gap_height = world->targets_distance + TARGET_AREAS_RADIUS - (world->arena_height / 2);

    if (gap_width > 0)
    {
        if ((world->target_areas[0].center.x >= 0) && (world->target_areas[0].center.x < gap_width))
            world->target_areas[0].center.x = gap_width;
        else if ((world->target_areas[0].center.x < 0) && (world->target_areas[0].center.x > (-gap_width)))
            world->target_areas[0].center.x = -gap_width;
    }

    if (gap_height > 0)
    {
        if ((world->target_areas[0].center.y >= 0) && (world->target_areas[0].center.y < gap_height))
            world->target_areas[0].center.y = gap_height;
        else if ((world->target_areas[0].center.y < 0) && (world->target_areas[0].center.y > (-gap_height)))
            world->target_areas[0].center.y = -gap_height;
    }

    float random_angle = random[world->id*NUM_WORLDS+(world->random_offset++)] * M_PI / 2;

    if ((world->target_areas[0].center.x > 0) && (world->target_areas[0].center.y > 0)) // first quadrant
        random_angle += M_PI;
    else if ((world->target_areas[0].center.x < 0) && (world->target_areas[0].center.y > 0)) // second quadrant
        random_angle += 3*M_PI/2;
    else if ((world->target_areas[0].center.x < 0) && (world->target_areas[0].center.y < 0)) // third quadrant
        random_angle += 0;
    else if ((world->target_areas[0].center.x > 0) && (world->target_areas[0].center.y < 0)) // third quadrant
        random_angle += M_PI/2;
    else // exactly center
        random_angle *= 4;

    world->target_areas[1].center.x = world->target_areas[0].center.x + cos(random_angle) * world->targets_distance;
    world->target_areas[1].center.y = world->target_areas[0].center.y + sin(random_angle) * world->targets_distance;
#endif

    unsigned int i, j, p = 0;

    for (i=0; i<NUM_ACTUATORS; i++)
    {
        for (j=0; j<(NUM_SENSORS+NUM_HIDDEN); j++)
            world->weights[i][j] = scale_param(params[p++], WEIGHTS_BOUNDARY_L, WEIGHTS_BOUNDARY_H);

        world->bias[i] = scale_param(params[p++], BIAS_BOUNDARY_L, BIAS_BOUNDARY_H);
    }

    for (i=0; i<NUM_HIDDEN; i++)
    {
        for (j=0; j<NUM_SENSORS; j++)
            world->weights_hidden[i][j] = scale_param(params[p++], WEIGHTS_BOUNDARY_L, WEIGHTS_BOUNDARY_H);

        world->bias_hidden[i] = scale_param(params[p++], BIAS_BOUNDARY_L, BIAS_BOUNDARY_H);
        world->timec_hidden[i] = scale_param(params[p++], TIMEC_BOUNDARY_L, TIMEC_BOUNDARY_H);
    }
}

void init_robot(__global float *random,
                __global world_t *world,
                __local transform_t *transforms,
                __global robot_t *robot)
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

    for (i=0; i<NUM_SENSORS; i++)
        robot->sensors[i] = 0;

    for (i=0; i<NUM_ACTUATORS; i++)
        robot->actuators[i] = 0;

    for (i=0; i<NUM_HIDDEN; i++)
        robot->hidden[i] = 0;

    set_random_position(random, world, transforms, robot);

#ifdef TEST
    if (robot->id == 0) {
        transforms[robot->id].pos.x = 0;
        transforms[robot->id].pos.y = 0;
        transforms[robot->id].rot.sin = 0;
        transforms[robot->id].rot.cos = 1;
    }
    else if (robot->id == 1) {
        transforms[robot->id].pos.x = 0.073;
        transforms[robot->id].pos.y = 0;
        transforms[robot->id].rot.sin = 1;
        transforms[robot->id].rot.cos = 0;
    }

    robot->previous_transform.pos.x = transforms[robot->id].pos.x;
    robot->previous_transform.pos.y = transforms[robot->id].pos.y;
    robot->previous_transform.rot.sin = transforms[robot->id].rot.sin;
    robot->previous_transform.rot.cos = transforms[robot->id].rot.cos;
#endif
}

void set_random_position(__global float *random,
                         __global world_t *world,
                         __local transform_t *transforms,
                         __global robot_t *robot)
{
    int otherid, i, collision = 1, tries = 0;

    while ((collision == 1) && (tries < 10))
    {
        float max_x = (world->arena_width / 2) - ROBOT_BODY_RADIUS;
        float max_y = (world->arena_height / 2) - ROBOT_BODY_RADIUS;
        float ra = random[world->id*NUM_WORLDS+(world->random_offset++)] * 2 * M_PI;

        transforms[robot->id].pos.x = (random[world->id*NUM_WORLDS+(world->random_offset++)] * 2 * max_x) - max_x;
        transforms[robot->id].pos.y = (random[world->id*NUM_WORLDS+(world->random_offset++)] * 2 * max_y) - max_y;
        transforms[robot->id].rot.sin = sin(ra);
        transforms[robot->id].rot.cos = cos(ra);

        robot->previous_transform.pos.x = transforms[robot->id].pos.x;
        robot->previous_transform.pos.y = transforms[robot->id].pos.y;
        robot->previous_transform.rot.sin = transforms[robot->id].rot.sin;
        robot->previous_transform.rot.cos = transforms[robot->id].rot.cos;

        collision = 0;

        // check for collision with other robots
        for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
        {
            if (robot->id != world->robots[otherid].id)
            {
                float dist = distance(transforms[robot->id].pos, transforms[otherid].pos);

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
            float dist = distance(transforms[robot->id].pos, world->target_areas[i].center);

            if (dist < world->target_areas[i].radius)
            {
                collision = 1;
                break;
            }
        }

        tries++;
    }
}

void step_world(__global world_t *world)
{
#if defined(MOVING_TARGETS) && (!defined(RANDOM_TARGET_AREAS))
    world->targets_angle += M_PI / (TA+TB);

    world->target_areas[0].center.x = cos(world->targets_angle) * (world->targets_distance / 2);
    world->target_areas[0].center.y = sin(world->targets_angle) * (world->targets_distance / 2);
    world->target_areas[1].center.x = -cos(world->targets_angle) * (world->targets_distance / 2);
    world->target_areas[1].center.y = -sin(world->targets_angle) * (world->targets_distance / 2);
#endif
}

void step_actuators(__global world_t *world, __local transform_t *transforms, __global robot_t *robot)
{
    robot->previous_transform.pos.x = transforms[robot->id].pos.x;
    robot->previous_transform.pos.y = transforms[robot->id].pos.y;
    robot->previous_transform.rot.sin = transforms[robot->id].rot.sin;
    robot->previous_transform.rot.cos = transforms[robot->id].rot.cos;

    robot->front_led = (robot->actuators[OUT_front_led] > 0.5) ? 1 : 0;
    robot->rear_led = (robot->actuators[OUT_rear_led] > 0.5) ? 1 : 0;

    robot->wheels_angular_speed.s0 = (robot->actuators[OUT_wheels0] * 2 * WHEELS_MAX_ANGULAR_SPEED) - WHEELS_MAX_ANGULAR_SPEED;
    robot->wheels_angular_speed.s1 = (robot->actuators[OUT_wheels1] * 2 * WHEELS_MAX_ANGULAR_SPEED) - WHEELS_MAX_ANGULAR_SPEED;

    int v1 = round(robot->actuators[OUT_wheels1] * (MOTOR_SAMPLE_COUNT - 1));
    int v2 = round(robot->actuators[OUT_wheels0] * (MOTOR_SAMPLE_COUNT - 1));

    transforms[robot->id].pos.x += MOTOR_LINEAR_SPEED_SAMPLES[v1][v2] * transforms[robot->id].rot.cos * TIME_STEP;
    transforms[robot->id].pos.y += MOTOR_LINEAR_SPEED_SAMPLES[v1][v2] * transforms[robot->id].rot.sin * TIME_STEP;

    float angle_robot = angle(transforms[robot->id].rot.sin, transforms[robot->id].rot.cos);
    angle_robot += MOTOR_ANGULAR_SPEED_SAMPLES[v1][v2] * TIME_STEP;
    transforms[robot->id].rot.sin = sin(angle_robot);
    transforms[robot->id].rot.cos = cos(angle_robot);
}

void step_sensors(__global world_t *world, __local transform_t *transforms, __global robot_t *robot)
{
    unsigned int i, j, otherid;
    __global robot_t *other;

    float2 pos = { transforms[robot->id].pos.x,
                   transforms[robot->id].pos.y };

    for (i=0; i<NUM_SENSORS; i++)
        robot->sensors[i] = 0;

    robot->entered_new_target_area = 0;

    fill_raycast_table(world, transforms, robot);

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

                float diff_angle = angle_rot(transforms[robot->id].rot) - wall_angle;

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
        other = &world->robots[otherid];

        float dist = distance(transforms[robot->id].pos, transforms[other->id].pos);

        if (robot->id == other->id)
            continue;

        if (dist < (2*ROBOT_BODY_RADIUS))
            robot->collision = 1;

        // IR against other robots
        float d = dist - 2*ROBOT_BODY_RADIUS;

        if (d < IR_ROUND_DIST_MAX)
        {
            int dist_idx;
            if (d <= IR_ROUND_DIST_MIN)
                dist_idx = 0;
            else
                dist_idx = (int) floor((d - IR_ROUND_DIST_MIN) / IR_ROUND_DIST_INTERVAL);

            float s = transforms[other->id].pos.y - transforms[robot->id].pos.y;
            float c = transforms[other->id].pos.x - transforms[robot->id].pos.x;
            float diff_angle = angle_rot(transforms[robot->id].rot) - angle(s, c);

            if (diff_angle >= (2*M_PI))
                diff_angle -= 2*M_PI;
            else if (diff_angle < 0)
                diff_angle += 2*M_PI;

            int angle_idx = (int) floor(diff_angle / ((2*M_PI) / IR_ROUND_ANGLE_COUNT));

            for (j = 0; j < 8; j++)
                robot->sensors[j] += IR_ROUND_SAMPLES[dist_idx][angle_idx][j] / 1024.0f;
        }

        // camera
        if ( (other->front_led == 1) && ((robot->raycast_table[other->id] & 1) != 0) )
            robot->sensors[IN_camera0] += 1;

        if ( (other->front_led == 1) && ((robot->raycast_table[other->id] & 2) != 0) )
            robot->sensors[IN_camera1] += 1;

        if ( (other->rear_led == 1) && ((robot->raycast_table[other->id] & 4) != 0) )
            robot->sensors[IN_camera2] += 1;

        if ( (other->rear_led == 1) && ((robot->raycast_table[other->id] & 8) != 0) )
            robot->sensors[IN_camera3] += 1;
    }

    for (i = 0; i < 2; i++)
    {
        float dist = distance(transforms[robot->id].pos, world->target_areas[i].center);

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

void step_collisions(__global world_t *world, __local transform_t *transforms, __global robot_t *robot)
{
    if (robot->collision != 0)
    {
        robot->collision = 0;

        transforms[robot->id].pos.x = robot->previous_transform.pos.x;
        transforms[robot->id].pos.y = robot->previous_transform.pos.y;
        transforms[robot->id].rot.sin = robot->previous_transform.rot.sin;
        transforms[robot->id].rot.cos = robot->previous_transform.rot.cos;

        robot->wheels_angular_speed.s0 = 0;
        robot->wheels_angular_speed.s1 = 0;
    }
}

void step_controllers(__global world_t *world, __local transform_t *transforms, __global robot_t *robot)
{
    unsigned int s,h,a;
    float aux;

    for (h=0; h<NUM_HIDDEN; h++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += world->weights_hidden[h][s] * robot->sensors[s];

        aux += world->bias_hidden[h];

        robot->hidden[h] = (world->timec_hidden[h] * robot->hidden[h]) + ((1 - world->timec_hidden[h]) * sigmoid(aux));
    }

    for (a=0; a<NUM_ACTUATORS; a++)
    {
        aux = 0;

        for (s=0; s<NUM_SENSORS; s++)
            aux += world->weights[a][s] * robot->sensors[s];

        for (h=0; h<NUM_HIDDEN; h++)
            aux += world->weights[a][NUM_SENSORS+h] * robot->hidden[h];

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

void fill_raycast_table(__global world_t *world, __local transform_t *transforms, __global robot_t *robot)
{
    unsigned int otherid, interid, targetid;
    float2 interpos, targetpos;

    float2 front = {ROBOT_BODY_RADIUS+0.005, 0};
    float2 rear = {-ROBOT_BODY_RADIUS-0.005, 0};

    float2 camerapos = transform_mul_vec(transforms[robot->id], front);

    float robot_angle = angle_rot(transforms[robot->id].rot);

    for (otherid = 0; otherid < ROBOTS_PER_WORLD; otherid++)
    {
        robot->raycast_table[otherid] = 0;

        float2 front_ray = transform_mul_vec(transforms[otherid], front) - camerapos;
        float front_ray_length = length(front_ray);
        float2 front_ray_normal = front_ray / front_ray_length;
        float front_angle = angle(front_ray_normal.y, front_ray_normal.x) - robot_angle;

        float2 rear_ray = transform_mul_vec(transforms[otherid], rear) - camerapos;
        float rear_ray_length = length(rear_ray);
        float2 rear_ray_normal = rear_ray / rear_ray_length;
        float rear_angle = angle(rear_ray_normal.y, rear_ray_normal.x) - robot_angle;

        if ( ((fabs(front_angle) > (CAMERA_ANGLE/2)) || (front_ray_length > CAMERA_RADIUS)) &&
             ((fabs(rear_angle) > (CAMERA_ANGLE/2)) || (rear_ray_length > CAMERA_RADIUS)) )
            continue;

        unsigned int front_intercept = 0, rear_intercept = 0;

        for (interid = 0; interid < ROBOTS_PER_WORLD; interid++)
        {
            interpos = transforms[interid].pos;

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

        if ((fabs(front_angle) <= (CAMERA_ANGLE/2)) && (front_ray_length <= CAMERA_RADIUS) && (front_intercept == 0))
        {
            if (front_angle <= 0)
                robot->raycast_table[otherid] |= 1;
            if (front_angle >= 0)
                robot->raycast_table[otherid] |= 2;
        }

        if ((fabs(rear_angle) <= (CAMERA_ANGLE/2)) && (rear_ray_length <= CAMERA_RADIUS) && (rear_intercept == 0))
        {
            if (rear_angle <= 0)
                robot->raycast_table[otherid] |= 4;
            if (rear_angle >= 0)
                robot->raycast_table[otherid] |= 8;
        }
    }

    for (targetid = 0; targetid < 2; targetid++)
    {
        targetpos = world->target_areas[targetid].center;

        robot->raycast_table[ROBOTS_PER_WORLD+targetid] = 0;

        float2 ray = targetpos - camerapos;
        float ray_length = length(ray);
        float2 ray_normal = ray / ray_length;
        float target_angle = angle(ray_normal.y, ray_normal.x) - robot_angle;

        if ( (fabs(target_angle) > (CAMERA_ANGLE/2)) || (ray_length > CAMERA_RADIUS) )
            continue;

        unsigned int intercept = 0;

        for (interid = 0; interid < ROBOTS_PER_WORLD; interid++)
        {
            interpos = transforms[interid].pos;

            float2 v1 = interpos - camerapos;

            float t = dot(v1, ray_normal);
            float2 proj = (ray_normal * t) + camerapos;

            intercept += ((t > 0) && (t < ray_length) &&
                          (distance(proj, interpos) < ROBOT_BODY_RADIUS)) ? 1 : 0;
        }

        if ((fabs(target_angle) <= (CAMERA_ANGLE/2)) && (ray_length <= CAMERA_RADIUS) && (intercept == 0))
        {
            if (target_angle <= 0)
                robot->raycast_table[ROBOTS_PER_WORLD+targetid] |= 4;
            if (target_angle >= 0)
                robot->raycast_table[ROBOTS_PER_WORLD+targetid] |= 8;
        }
    }
}

#endif
