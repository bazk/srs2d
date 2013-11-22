#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <CL/opencl.h>

#define NUM_WORLDS 1
#define ROBOTS_PER_WORLD 10
#define ANN_PARAM_SIZE 113
#define TARGETS_DISTANCE 0.7

#define TIME_STEP 0.1
#define TA 18600
#define TB 5400

#define NUM_SENSORS    13
#define NUM_ACTUATORS   4
#define NUM_HIDDEN      3

typedef struct {
    cl_float sin;
    cl_float cos;
} rotation_t;

typedef struct {
    cl_float2 pos;
    rotation_t rot;
} transform_t;

typedef struct {
    cl_float2 p1;
    cl_float2 p2;
} wall_t;

typedef struct {
    cl_uint id;
    transform_t transform;
    transform_t previous_transform;
    cl_float2 wheels_angular_speed;
    cl_int front_led;
    cl_int rear_led;
    cl_int collision;
    cl_float energy;
    cl_float fitness;
    cl_int last_target_area;
    cl_int entered_new_target_area;

    cl_float sensors[NUM_SENSORS];
    cl_float actuators[NUM_ACTUATORS];
    cl_float hidden[NUM_HIDDEN];
} robot_t;

typedef struct {
    cl_float2 center;
    cl_float radius;
} target_area_t;

typedef struct {
    cl_uint id;

    robot_t robots[ROBOTS_PER_WORLD];

    cl_float k;

    cl_float arena_height;
    cl_float arena_width;

    wall_t walls[4];

    cl_float targets_distance;

    target_area_t target_areas[2];

    // ANN parameters
    cl_float weights[NUM_ACTUATORS*(NUM_SENSORS+NUM_HIDDEN)];
    cl_float bias[NUM_ACTUATORS];

    cl_float weights_hidden[NUM_HIDDEN*NUM_SENSORS];
    cl_float bias_hidden[NUM_HIDDEN];
    cl_float timec_hidden[NUM_HIDDEN];
} world_t;

inline void assert(int cond, const char * desc)
{
    if (!cond) {
        fprintf(stderr, "ERROR: %s\n", desc);
        exit(EXIT_FAILURE);
    }
}

size_t load_source(const char* filename, char **source)
{
    FILE *file;
    size_t length;

    file = fopen(filename, "rb");
    if (file == 0)
        return -1;

    // get the length of the source code
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    fseek(file, 0, SEEK_SET);

    // allocate a buffer for the source code string and read it in
    (*source) = (char*) malloc(length+1);

    if (fread((*source), length, 1, file) != 1)
    {
        fclose(file);
        free((*source));
        return -1;
    }

    (*source)[length] = '\0';

    fclose(file);

    return length;
}

void execute(cl_device_id device, cl_context context, cl_command_queue queue)
{
    cl_int err;

    cl_program program;
    cl_kernel kernel;

    char *source;
    size_t length;

    size_t global_size[] = {NUM_WORLDS, ROBOTS_PER_WORLD};
    size_t local_size[] = {NUM_WORLDS, ROBOTS_PER_WORLD};

    cl_uint ann_param_size = ANN_PARAM_SIZE;
    cl_float targets_distance = TARGETS_DISTANCE;
    cl_uint save = 0;

    char build_options[4096];
    sprintf(
        build_options,
        "-I\"%s\" -DNUM_WORLDS=%d -DROBOTS_PER_WORLD=%d -DTIME_STEP=%f -DTA=%d -DTB=%d -DWORLDS_PER_LOCAL=%d -DROBOTS_PER_LOCAL=%d",
        "/home/buratti/srs2d/srs2d/kernels", NUM_WORLDS, ROBOTS_PER_WORLD, TIME_STEP, TA, TB, (int) local_size[0],  (int) local_size[1]
    );

    // load source code from file
    length = load_source("../srs2d/kernels/physics.cl", &source);
    assert(length > 0, "load_source()");

    // create and build program
    program = clCreateProgramWithSource(context, 1, (const char **) &source, &length, &err);
    assert(err == CL_SUCCESS, "clCreateProgramWithSource()");

    err = clBuildProgram(program, 0, NULL, (const char*) build_options, NULL, NULL);
    if ((err != CL_SUCCESS) && (err == CL_BUILD_PROGRAM_FAILURE))
    {
        cl_build_status build_status;
        char *build_log;
        size_t build_log_size;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
        assert(err == CL_SUCCESS, "clGetProgramBuildInfo()");

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        assert(err == CL_SUCCESS, "clGetProgramBuildInfo()");

        build_log = (char *) malloc(build_log_size+1);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
        assert(err == CL_SUCCESS, "clGetProgramBuildInfo()");

        build_log[build_log_size] = '\0';

        fprintf(stderr, "BUILD LOG:\n%s\n\n", build_log);

        free(build_log);
        exit(EXIT_FAILURE);
    }

    // create buffers
    cl_mem ranluxcl_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 112 * NUM_WORLDS * ROBOTS_PER_WORLD, NULL, &err);
    assert(err == CL_SUCCESS, "clCreateBuffer()");

    cl_mem worlds_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(world_t) * NUM_WORLDS, NULL, &err);
    assert(err == CL_SUCCESS, "clCreateBuffer()");

    cl_mem param_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_char) * ANN_PARAM_SIZE * NUM_WORLDS, NULL, &err);
    assert(err == CL_SUCCESS, "clCreateBuffer()");

    cl_mem fitness_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * NUM_WORLDS, NULL, &err);
    assert(err == CL_SUCCESS, "clCreateBuffer()");

    // create kernel
    kernel = clCreateKernel(program, "simulate", &err);
    assert(err == CL_SUCCESS, "clCreateKernel()");

    err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &ranluxcl_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &worlds_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_float), (void*) &targets_distance);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &param_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), (void*) &ann_param_size);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &fitness_buffer);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 10, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 11, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 12, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 13, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 14, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 15, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(kernel, 16, sizeof(cl_uint), (void*) &save);
    assert(err == CL_SUCCESS, "clSetKernelArg()");

    // execute kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(err == CL_SUCCESS, "clEnqueueNDRangeKernel()");

    // copy results from device
    float *fitness = (float *) malloc(NUM_WORLDS * sizeof(cl_float));
    err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, NUM_WORLDS * sizeof(cl_float), fitness, 0, NULL, NULL);
    assert(err == CL_SUCCESS, "clEnqueueReadBuffer()");

    printf("fitness = %f\n", fitness[0]);

    free(fitness);
    free(source);
}

int main(int argc, char **argv)
{
    cl_int err;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // get an OpenCL platform
    err = clGetPlatformIDs(1, &platform, NULL);
    assert(err == CL_SUCCESS, "clGetPlatformIDs()");

    // get devices
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    assert(err == CL_SUCCESS, "clGetDeviceIDs()");

    // create context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS, "clCreateContext()");

    // create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    assert(err == CL_SUCCESS, "clCreateCommandQueue()");

    execute(device, context, queue);

    exit(EXIT_SUCCESS);
}