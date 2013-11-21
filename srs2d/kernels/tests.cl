#ifndef __TESTS_CL__
#define __TESTS_CL__

/* int _raycast(__global world_t *worlds)
{
    float2 front = {(ROBOT_BODY_RADIUS + LED_PROTUBERANCE), 0};

    float2 orig = worlds[0].robots[0].transform.pos;
    float angle_robot = angle(worlds[0].robots[0].transform.rot.sin, worlds[0].robots[0].transform.rot.cos);

    float2 dest = transform_mul_vec(worlds[0].robots[2].transform, front);
    float d = distance(orig, dest);

    if (d < CAMERA_RADIUS)
    {
        float angle_dest = angle(dest.y - orig.y, dest.x - orig.x);

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
}*/

#endif