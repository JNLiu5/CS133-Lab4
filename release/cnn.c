#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>
#include "cnn.h"
#include "kernel_cl.h"

#define LOCAL_SIZE 4
#define GLOBAL_SIZE 256

inline void checkErr(cl_int err, const char * name) {
   if (err != CL_SUCCESS) {
      fprintf(stderr, "ERROR: %s (%d)\n", name, err);
      exit(EXIT_FAILURE);
   }
}

// Sequential CNN implementation
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	static float C[NUM][IMROW][IMROW];

	for(int i = 0; i < NUM; i++) {
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++)
				C[i][h][w] = bias[i];
		}
	}

// Convolution
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < NUM; j++) {
			for(int h = 0; h < IMROW; h++) {
				for(int w = 0; w < IMROW; w++) {
					for(int p = 0; p < KERNEL; p++) {
						for(int q = 0; q < KERNEL; q++)
							C[i][h][w] += weight[i][j][p][q] * Cin[j][h + p][w + q];
					}
				}
			}
		}
	}

// ReLU
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < IMROW; h++) {
			for (int w = 0; w < IMROW; w++) {
				C[i][h][w] = fmax(0, C[i][h][w]);
			}	
		}
	}

// Max pooling
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < OUTIMROW; h++) {
			for (int w = 0; w < OUTIMROW; w++) {
				float local_max = C[i][2 * h][2 * w];
				local_max = fmax(local_max, C[i][2 * h + 1][2 * w]);
				local_max = fmax(local_max, C[i][2 * h + 1][2 * w + 1]);
				local_max = fmax(local_max, C[i][2 * h][2 * w + 1]);
				Cout[i][h][w] = local_max;
			}
		}
	}
}

int main(){
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	fprintf(stderr, "Start cnn computation\n");
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// --- Please add OpenCL setup code below ---
	cl_int status;

	cl_uint num_platforms = 0;
	status = clGetPlatformIDs(0, NULL, &num_platforms);
	checkErr(status, "Retrieve the number of platforms");
	printf("num_platforms: %d\n", num_platforms);

	cl_platform_id* platforms = NULL;
	platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	printf("platforms: %d\n", platforms);

	status = clGetPlatformIDs(num_platforms, platforms, NULL);
	checkErr(status, "Fill in the platforms");

	int platform_index = -1;
	int i;
	for(i = 0; i < num_platforms; i++) {
		char vendor[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
		char vendorF[7];
		memcpy((void*)vendorF, (void*)vendor, 6);
		vendorF[6] = '\0';
		fprintf(stderr, "%s\n", vendorF);
		if(strcmp(vendorF, "NVIDIA") == 0) {
			platform_index = i;
			break;
		}
	}
	if(platform_index == -1) {
		printf("GPU platform not found!\n");
		exit(1);
	}

	cl_uint num_devices = 0;
	status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	checkErr(status, "Retrieve the number of devices");
	printf("#devices: %d, status %d\n", num_devices, status);

	cl_device_id* devices;
	devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
	printf("devices: %d\n", devices);

	status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	checkErr(status, "Fill in the devices");

	cl_context context;
	context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &status);

	cl_command_queue cmd_queue;
	cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);

	cl_mem buf_Cout;
	size_t cout_size = sizeof(float) * NUM * OUTIMROW * OUTIMROW;
	buf_Cout = clCreateBuffer(context, CL_MEM_READ_ONLY, cout_size, NULL, &status);
	
	cl_mem buf_Cin;
	size_t cin_size = sizeof(float) * NUM * INIMROW * INIMROW;
	buf_Cin = clCreateBuffer(context, CL_MEM_READ_ONLY, cin_size, NULL, &status);

	cl_mem buf_weight;
	size_t weight_size = sizeof(float) * NUM * NUM * KERNEL * KERNEL;
	buf_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, &status);
	
	cl_mem buf_bias;
	size_t bias_size = sizeof(float) * NUM;
	buf_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, &status);

	status = clEnqueueWriteBuffer(cmd_queue, buf_Cin, CL_FALSE, 0, cin_size, Cin, 0, NULL, NULL);
	checkErr(status, "Write buffer Cin");

	status = clEnqueueWriteBuffer(cmd_queue, buf_weight, CL_FALSE, 0, weight_size, weight, 0, NULL, NULL);
	checkErr(status, "Write buffer weight");

	status = clEnqueueWriteBuffer(cmd_queue, buf_bias, CL_FALSE, 0, bias_size, bias, 0, NULL, NULL);
	checkErr(status, "Write buffer bias");

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_cl, NULL, &status);

	status = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
	if(status == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		fprintf(stderr, "%s\n", log);
		exit(1);
	}

	cl_kernel kernel;
	kernel = clCreateKernel(program, "conv", &status);

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_Cout);
	checkErr(status, "Set Arg Cout");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_Cin);
	checkErr(status, "Set Arg Cin");
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_weight);
	checkErr(status, "Set Arg weight");
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_bias);
	checkErr(status, "Set Arg bias");

	size_t local[1] = {LOCAL_SIZE};
	size_t global[1] = {GLOBAL_SIZE};

	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
	checkErr(status, "Execute kernel");

   // Run the sequential implementation for now. 
   // You should replace this with a call to your kernel
//	conv(Cout, Cin, weight, bias);	

   // --- Timing stuff
	gettimeofday(&t2, NULL);
	float elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

   // Please disable the error check before handing in your submission
   // Reminder: We will be measuring your performance externally! (using a unix time invocation)
	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queue);
	clReleaseMemObject(buf_Cout);
	clReleaseMemObject(buf_Cin);
	clReleaseMemObject(buf_weight);
	clReleaseMemObject(buf_bias);
	clReleaseContext(context);

	free(platforms);
	free(devices);

	return 0;
}
