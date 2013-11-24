#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <CL/opencl.h>

#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

#define NUM_WORLDS 120
#define ROBOTS_PER_WORLD 10
#define ANN_PARAM_SIZE 113
#define TARGETS_DISTANCE 1.1

#define TIME_STEP 0.1
#define TA 600
#define TB 5400

#define NUM_SENSORS    13
#define NUM_ACTUATORS   4
#define NUM_HIDDEN      3

char params[ANN_PARAM_SIZE] = {
    0x66, 0x74, 0x13, 0x06, 0x25, 0x6f, 0x0b, 0xd5, 0x8c, 0xad, 0x8a, 0x7d,
    0x87, 0x63, 0x91, 0xd4, 0x98, 0x0d, 0x29, 0xef, 0x7e, 0xb9, 0x50, 0xf8,
    0x1e, 0x2e, 0x60, 0x35, 0x90, 0xc7, 0x69, 0xe5, 0x6c, 0xc9, 0x96, 0xe0,
    0xa5, 0x19, 0x41, 0x3a, 0xa0, 0xe9, 0x5c, 0x3b, 0xae, 0x78, 0xe1, 0xd4,
    0x3b, 0xb2, 0xdf, 0x06, 0x54, 0xfd, 0x86, 0x5c, 0x62, 0x41, 0xc3, 0x73,
    0x16, 0xcf, 0xf6, 0xd1, 0xd1, 0xf9, 0x66, 0x43, 0x42, 0x13, 0x63, 0xdd,
    0xc6, 0x15, 0x54, 0x0d, 0x78, 0x82, 0x8e, 0x58, 0xc5, 0x36, 0x02, 0x2f,
    0x4e, 0x9b, 0x31, 0x4f, 0x56, 0xb0, 0x5a, 0xa3, 0x53, 0xef, 0x47, 0xa1,
    0x0f, 0xf8, 0xd7, 0x0b, 0xae, 0xaa, 0x19, 0x35, 0xc0, 0x77, 0x8e, 0x37,
    0x38, 0x70, 0x4e, 0x67, 0xa6
};

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

const char *get_error_string(cl_int err)
{
    switch (err)
    {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";

    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    default: return "Unknown OpenCL error";
    }
}

inline void assert(int cond, cl_int err, const char * desc)
{
    if (!cond) {
        fprintf(stderr, "ERROR: %s %s\n", get_error_string(err), desc);
        exit((int) err);
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
    size_t local_size[] = {6, ROBOTS_PER_WORLD};

    cl_uint ann_param_size = ANN_PARAM_SIZE;
    cl_float targets_distance = TARGETS_DISTANCE;
    cl_uint save = 0;

    cl_uint seed = rand();

    char *param_list = (char*) malloc(NUM_WORLDS * ANN_PARAM_SIZE);
    unsigned int i, j;
    for (i=0; i<NUM_WORLDS; i++)
    {
        for (j=0; j < ANN_PARAM_SIZE; j++)
        {
            param_list[(i*ANN_PARAM_SIZE)+j] = params[j];
        }
    }

    char build_options[4096];
    sprintf(
        build_options,
        "-I\"%s\" -DNUM_WORLDS=%d -DROBOTS_PER_WORLD=%d -DTIME_STEP=%f -DTA=%d -DTB=%d -DWORLDS_PER_LOCAL=%d -DROBOTS_PER_LOCAL=%d",
        "/home/buratti/srs2d/srs2d/kernels", NUM_WORLDS, ROBOTS_PER_WORLD, TIME_STEP, TA, TB, (int) local_size[0],  (int) local_size[1]
    );

    // load source code from file
    length = load_source("../srs2d/kernels/physics.cl", &source);
    assert(length > 0, -1, "load_source()");

    // create and build program
    program = clCreateProgramWithSource(context, 1, (const char **) &source, &length, &err);
    assert(err == CL_SUCCESS, err, "clCreateProgramWithSource()");

    err = clBuildProgram(program, 0, NULL, (const char*) build_options, NULL, NULL);
    if ((err != CL_SUCCESS) && (err == CL_BUILD_PROGRAM_FAILURE))
    {
        cl_build_status build_status;
        char *build_log;
        size_t build_log_size;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
        assert(err == CL_SUCCESS, err, "clGetProgramBuildInfo()");

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        assert(err == CL_SUCCESS, err, "clGetProgramBuildInfo()");

        build_log = (char *) malloc(build_log_size+1);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
        assert(err == CL_SUCCESS, err, "clGetProgramBuildInfo()");

        build_log[build_log_size] = '\0';

        fprintf(stderr, "BUILD LOG:\n%s\n\n", build_log);

        free(build_log);
        exit(EXIT_FAILURE);
    }

    // create buffers
    cl_mem ranluxcl_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 112 * NUM_WORLDS * ROBOTS_PER_WORLD, NULL, &err);
    assert(err == CL_SUCCESS, err, "clCreateBuffer()");

    cl_mem worlds_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(world_t) * NUM_WORLDS, NULL, &err);
    assert(err == CL_SUCCESS, err, "clCreateBuffer()");

    cl_mem param_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * ANN_PARAM_SIZE * NUM_WORLDS, NULL, &err);
    assert(err == CL_SUCCESS, err, "clCreateBuffer()");

    cl_mem fitness_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * NUM_WORLDS, NULL, &err);
    assert(err == CL_SUCCESS, err, "clCreateBuffer()");

    // copy param_list to device
    err = clEnqueueWriteBuffer(queue, param_buffer, CL_TRUE, 0, sizeof(cl_char) * ANN_PARAM_SIZE * NUM_WORLDS, param_list, 0, NULL, NULL);
    assert(err == CL_SUCCESS, err, "clEnqueueWriteBuffer()");

    // execute init_ranluxcl kernel
    kernel = clCreateKernel(program, "init_ranluxcl", &err);
    assert(err == CL_SUCCESS, err, "clCreateKernel()");

    err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_uint), (void*) &seed);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &ranluxcl_buffer);
    assert(err == CL_SUCCESS, err, "clSetKernelArg()");

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(err == CL_SUCCESS, err, "clEnqueueNDRangeKernel()");

    err = clReleaseKernel(kernel);
    assert(err == CL_SUCCESS, err, "clReleaseKernel()");

    // execute simulate kernel
    kernel = clCreateKernel(program, "simulate", &err);
    assert(err == CL_SUCCESS, err, "clCreateKernel()");

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
    assert(err == CL_SUCCESS, err, "clSetKernelArg()");

    // execute kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    assert(err == CL_SUCCESS, err, "clEnqueueNDRangeKernel()");

    // copy results from device
    float *fitness = (float *) malloc(NUM_WORLDS * sizeof(cl_float));
    err = clEnqueueReadBuffer(queue, fitness_buffer, CL_TRUE, 0, NUM_WORLDS * sizeof(cl_float), fitness, 0, NULL, NULL);
    assert(err == CL_SUCCESS, err, "clEnqueueReadBuffer()");

    for (i=0; i<NUM_WORLDS; i++)
        printf("fitness[%d] = %f\n", i, fitness[i]);

    err = CL_SUCCESS;
    err |= clReleaseMemObject(ranluxcl_buffer);
    err |= clReleaseMemObject(worlds_buffer);
    err |= clReleaseMemObject(param_buffer);
    err |= clReleaseMemObject(fitness_buffer);
    assert(err == CL_SUCCESS, err, "clReleaseMemObject()");

    err = clReleaseKernel(kernel);
    assert(err == CL_SUCCESS, err, "clReleaseKernel()");

    err = clReleaseProgram(program);
    assert(err == CL_SUCCESS, err, "clReleaseProgram()");

    free(param_list);
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

    srand(time(NULL));

    // get an OpenCL platform
    err = clGetPlatformIDs(1, &platform, NULL);
    assert(err == CL_SUCCESS, err, "clGetPlatformIDs()");

    // get devices
    err = clGetDeviceIDs(platform, DEVICE_TYPE, 1, &device, NULL);
    assert(err == CL_SUCCESS, err, "clGetDeviceIDs()");

    // create context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS, err, "clCreateContext()");

    // create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    assert(err == CL_SUCCESS, err, "clCreateCommandQueue()");

    execute(device, context, queue);

    err = clReleaseCommandQueue(queue);
    assert(err == CL_SUCCESS, err, "clReleaseCommandQueue()");

    err = clReleaseContext(context);
    assert(err == CL_SUCCESS, err, "clReleaseContext()");

    exit(EXIT_SUCCESS);
}