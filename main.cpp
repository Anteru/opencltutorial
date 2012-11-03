#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

std::string GetPlatformName (cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

std::string GetDeviceName (cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetDeviceInfo (id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program CreateProgram (const std::string& source,
	cl_context context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

int main ()
{
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetPlatformIDs.html
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		deviceIds.data (), nullptr, nullptr, &error);
	CheckError (error);

	std::cout << "Context created" << std::endl;

	cl_program program = CreateProgram (LoadKernel ("kernels/saxpy.cl"),
		context);

	CheckError (clBuildProgram (program, deviceIdCount, deviceIds.data (), nullptr, nullptr, nullptr));

	cl_kernel kernel = clCreateKernel (program, "SAXPY", &error);
	CheckError (error);

	// Prepare some test data
	static const size_t testDataSize = 1 << 10;
	std::vector<float> a (testDataSize), b (testDataSize);
	for (int i = 0; i < testDataSize; ++i) {
		a [i] = static_cast<float> (23 ^ i);
		b [i] = static_cast<float> (42 ^ i);
	}

	cl_mem aBuffer = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof (float) * (testDataSize),
		a.data (), &error);
	CheckError (error);

	cl_mem bBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (float) * (testDataSize),
		b.data (), &error);
	CheckError (error);

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateCommandQueue.html
	cl_command_queue queue = clCreateCommandQueue (context, deviceIds [0],
		0, &error);
	CheckError (error);

	clSetKernelArg (kernel, 0, sizeof (cl_mem), &aBuffer);
	clSetKernelArg (kernel, 1, sizeof (cl_mem), &bBuffer);
	static const float two = 2.0f;
	clSetKernelArg (kernel, 2, sizeof (float), &two);

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
	const size_t globalWorkSize [] = { testDataSize, 0, 0 };
	CheckError (clEnqueueNDRangeKernel (queue, kernel, 1,
		nullptr,
		globalWorkSize,
		nullptr,
		0, nullptr, nullptr));

	// Get the results back to the host
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReadBuffer.html
	CheckError (clEnqueueReadBuffer (queue, bBuffer, CL_TRUE, 0,
		sizeof (float) * testDataSize,
		b.data (),
		0, nullptr, nullptr));

	clReleaseCommandQueue (queue);

	clReleaseMemObject (bBuffer);
	clReleaseMemObject (aBuffer);

	clReleaseKernel (kernel);
	clReleaseProgram (program);

	clReleaseContext (context);
}