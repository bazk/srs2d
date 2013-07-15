typedef struct {
    float2 pos;
    float2 rot;
} transform_t;

typedef struct {
    transform_t transform;
    float2 linear_velocity;
    float angular_speed;
} body_t;

typedef struct {
    float2 pos;
    float2 linear_velocity;
    float angular_speed;
} body_t;

__kernel void size_of_body_t(__global int *result)
{
    *result = (int) sizeof(body_t);
}

__kernel void set_bodies(__global body_t *bodies, __global const float2 *pos, __global const float2 *rot, __global const float2 *linear_velocity, __global const float *angular_speed)
{
    int gid = get_global_id(0);

    bodies[gid].transform.pos = pos[gid];
    bodies[gid].transform.rot = rot[gid];
    bodies[gid].linear_velocity = linear_velocity[gid];
    bodies[gid].angular_speed = angular_speed[gid];
}

__kernel void get_bodies(__global const body_t *bodies, __global float2 *pos, __global float2 *rot, __global float2 *linear_velocity, __global float *angular_speed)
{
    int gid = get_global_id(0);

    pos[gid] = bodies[gid].transform.pos;
    rot[gid] = bodies[gid].transform.rot;
    linear_velocity[gid] = bodies[gid].linear_velocity;
    angular_speed[gid] = bodies[gid].angular_speed;
}

__kernel void step_dynamics(__global body_t *bodies)
{
    int gid = get_global_id(0);

    bodies[gid].transform.pos.x += bodies[gid].linear_velocity.x;
    bodies[gid].transform.pos.y += bodies[gid].linear_velocity.y;

    float angle = atan2(bodies[gid].transform.rot.x, bodies[gid].transform.rot.y) + bodies[gid].angular_speed;
    bodies[gid].transform.rot.x = sin(angle);
    bodies[gid].transform.rot.y = cos(angle);
}