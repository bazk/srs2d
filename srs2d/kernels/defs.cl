#ifndef __DEFS_CL__
#define __DEFS_CL__

#define ROBOT_BODY_RADIUS           0.035
#define WHEELS_MAX_ANGULAR_SPEED    2.46
#define WHEELS_DISTANCE             0.053
#define WHEELS_RADIUS               0.02
#define CAMERA_RADIUS               0.35
#define CAMERA_ANGLE                2.5132741228718345 // 144 degrees (72+72)
#define TARGET_AREAS_RADIUS         0.16

#define ARENA_HEIGHT    2.5
#define ARENA_WIDTH_MIN 2.5
#define ARENA_WIDTH_MAX 2.9

#define NUM_SENSORS    13
#define NUM_ACTUATORS   4
#define NUM_HIDDEN      3

#define IN_proximity0   0
#define IN_proximity1   1
#define IN_proximity2   2
#define IN_proximity3   3
#define IN_proximity4   4
#define IN_proximity5   5
#define IN_proximity6   6
#define IN_proximity7   7
#define IN_camera0      8
#define IN_camera1      9
#define IN_camera2      10
#define IN_camera3      11
#define IN_ground       12

#define OUT_wheels0     0
#define OUT_wheels1     1
#define OUT_front_led   2
#define OUT_rear_led    3

#define HID_hidden0     0
#define HID_hidden1     1
#define HID_hidden2     2

#define WEIGHTS_BOUNDARY_L -5.0
#define WEIGHTS_BOUNDARY_H  5.0
#define BIAS_BOUNDARY_L    -5.0
#define BIAS_BOUNDARY_H     5.0
#define TIMEC_BOUNDARY_L    0.0
#define TIMEC_BOUNDARY_H    1.0

typedef struct {
    float sin;
    float cos;
} rotation_t;

typedef struct {
    float2 pos;
    rotation_t rot;
} transform_t;

typedef struct {
    float2 p1;
    float2 p2;
} wall_t;

typedef struct {
    unsigned int id;
    transform_t previous_transform;
    float2 wheels_angular_speed;
    unsigned int front_led;
    unsigned int rear_led;
    unsigned int collision;
    float energy;
    float fitness;
    int last_target_area;
    unsigned int entered_new_target_area;

    float sensors[NUM_SENSORS];
    float actuators[NUM_ACTUATORS];
    float hidden[NUM_HIDDEN];

    char raycast_table[ROBOTS_PER_WORLD+2]; // +2 'cause target areas
} robot_t;

typedef struct {
    float2 center;
    float radius;
} target_area_t;

typedef struct {
    unsigned int id;

    robot_t robots[ROBOTS_PER_WORLD];

    float arena_height;
    float arena_width;

    wall_t walls[4];
    target_area_t target_areas[2];

    // ANN parameters
    float weights[NUM_ACTUATORS][NUM_SENSORS+NUM_HIDDEN];
    float bias[NUM_ACTUATORS];

    float weights_hidden[NUM_HIDDEN][NUM_SENSORS];
    float bias_hidden[NUM_HIDDEN];
    float timec_hidden[NUM_HIDDEN];

    unsigned int random_offset;
} world_t;

#endif