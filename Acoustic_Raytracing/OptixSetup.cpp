// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "OptixSetup.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{

extern "C" char embedded_ptx_code[];
const double pi = 3.14159265358979323846;

// Approximation of the golden ratio from https://www2.cs.arizona.edu/icon/oddsends/phi.htm
const double phi = 1.61803398874989484820;

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void *data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void *data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	TriangleMeshSBTData data;
};

int next_pow_2(int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}
//! add aligned cube with front-lower-left corner and size
void TriangleMesh::addCube(const vec3f &center, const vec3f &size)
{
	PING;
	affine3f xfm;
	xfm.p = center - 0.5f * size;
	xfm.l.vx = vec3f(size.x, 0.f, 0.f);
	xfm.l.vy = vec3f(0.f, size.y, 0.f);
	xfm.l.vz = vec3f(0.f, 0.f, size.z);
	addUnitCube(xfm);
}

/*! add a unit cube (subject to given xfm matrix) to the current
			triangleMesh */
void TriangleMesh::addUnitCube(const affine3f &xfm)
{
	unsigned int firstVertexID = (unsigned int)vertex.size();
	vertex.push_back(xfmPoint(xfm, vec3f(0.f, 0.f, 0.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(1.f, 0.f, 0.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(0.f, 1.f, 0.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(1.f, 1.f, 0.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(0.f, 0.f, 1.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(1.f, 0.f, 1.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(0.f, 1.f, 1.f)));
	vertex.push_back(xfmPoint(xfm, vec3f(1.f, 1.f, 1.f)));

	int indices[] = {0, 1, 3, 2, 3, 0,
					 5, 7, 6, 5, 6, 4,
					 0, 4, 5, 0, 5, 1,
					 2, 3, 7, 2, 7, 6,
					 1, 5, 7, 1, 7, 3,
					 4, 0, 2, 4, 2, 6};
	for (int i = 0; i < 12; i++)
		index.push_back(firstVertexID + vec3ui(indices[3 * i + 0],
											   indices[3 * i + 1],
											   indices[3 * i + 2]));
}
/*
		Resources on icospheres:
		https://mathworld.wolfram.com/RegularIcosahedron.html
		Says that the coordinates for an square edge length of a = 2 results in coordinates of (0, +/- 1, +/- phi) 
		where phi = golden ratio

		https://en.wikipedia.org/wiki/Regular_icosahedron#Dimensions
		Quick way to find Rc = a * sin(2 * pi / 5)
		Same equation is also found in the mathworld link without the sine subsitution

		https://math.stackexchange.com/questions/441327/coordinates-of-icosahedron-vertices-with-variable-radius
		coordinates are perumatations of (0, +/- 1, +/- phi) * r / (2 * sin(2 * pi / 5)
		
		https://www.opengl.org.ru/docs/pg/0208.html
		When the radius = 1, the permutations become (0, +/-X, +/-Z) where 
		X = .525731112119133606;
		Z = .850650808352039932;

		which aligns with the math from the stack exchange
		*/
void TriangleMesh::addSphere(vec3f center, float radius, int recursionLevel)
{
	PING;
	affine3f xfm;
	xfm.p = center;
	xfm.l.vx = vec3f(radius, 0.f, 0.f);
	xfm.l.vy = vec3f(0.f, radius, 0.f);
	xfm.l.vz = vec3f(0.f, 0.f, radius);

	temp_vertex.clear();
	vertex.clear();
	index.clear();
	m_center = center;
	m_radius = radius;
	std::vector<vec3ui> local_index;
	float X = 1.0f / (2.0f * sin(2.0f * pi / 5.0f));
	float Z = phi / (2.0f * sin(2.0f * pi / 5.0f));
	const float N = 0.f;

	temp_vertex.push_back(vec3f(-X, N, Z));
	temp_vertex.push_back(vec3f(X, N, Z));
	temp_vertex.push_back(vec3f(-X, N, -Z));
	temp_vertex.push_back(vec3f(X, N, -Z));
	temp_vertex.push_back(vec3f(N, Z, X));
	temp_vertex.push_back(vec3f(N, Z, -X));
	temp_vertex.push_back(vec3f(N, -Z, X));
	temp_vertex.push_back(vec3f(N, -Z, -X));
	temp_vertex.push_back(vec3f(Z, X, N));
	temp_vertex.push_back(vec3f(-Z, X, N));
	temp_vertex.push_back(vec3f(Z, -X, N));
	temp_vertex.push_back(vec3f(-Z, -X, N));

	unsigned int indices[] = {
		0, 4, 1, 0, 9, 4,
		9, 5, 4, 4, 5, 8,
		4, 8, 1, 8, 10, 1,
		8, 3, 10, 5, 3, 8,
		5, 2, 3, 2, 7, 3,
		7, 10, 3, 7, 6, 10,
		7, 11, 6, 11, 0, 6,
		0, 1, 6, 6, 1, 10,
		9, 0, 11, 9, 11, 2,
		9, 2, 5, 7, 2, 11};
	for (int i = 0; i < 20; i++)
	{
		local_index.push_back(vec3ui(indices[3 * i + 0],
									 indices[3 * i + 1],
									 indices[3 * i + 2]));
	}

	// refine triangles
	for (int i = 0; i < recursionLevel; i++)
	{
		std::vector<vec3ui> faces2;
		for (int j = 0; j < local_index.size(); j++)
		{
			vec3ui tri_idx = local_index[j];
			// replace triangle by 4 triangles
			int a = getMiddlePoint(tri_idx.x, tri_idx.y);
			int b = getMiddlePoint(tri_idx.y, tri_idx.z);
			int c = getMiddlePoint(tri_idx.z, tri_idx.x);

			if (i == 0)
			{
				faces2.push_back(vec3ui(tri_idx.x, a, c));
				faces2.push_back(vec3ui(tri_idx.y, b, a));
				faces2.push_back(vec3ui(tri_idx.z, c, b));
				faces2.push_back(vec3ui(a, b, c));
			}
			else
			{
				//faces2.push_back(tri_idx);
				faces2.push_back(vec3ui(tri_idx.x, a, c));
				faces2.push_back(vec3ui(tri_idx.y, b, a));
				faces2.push_back(vec3ui(tri_idx.z, c, b));
				faces2.push_back(vec3ui(a, b, c));
				//if (j == 0 || j == 2 || j == 3) {
				//faces2.push_back(vec3ui(tri_idx.y, b, a));
				//faces2.push_back(vec3ui(tri_idx.z, c, b));
				//faces2.push_back(vec3ui(a, b, c));
				//}
			}

			//faces2.push_back(vec3ui(a, tri_idx.y, c));

			/*NOTES:
				Order 1:
					all 4 work
				Order 2:
					x a c
					x a c + y b a
					NOT x a c + y b a + z c b
					NOT x a c + y b a + a b c
					x a c + a b c
					y b a + a b c
					*/
		}
		local_index = faces2;
	}

	//add completed list of indices to the mesh
	for (int i = 0; i < local_index.size(); i++)
	{
		index.push_back(local_index[i]);
	}
	for (int i = 0; i < temp_vertex.size(); i++)
		vertex.push_back(xfmPoint(xfm, temp_vertex[i]));

	//vec3ui max = *(std::max(index.begin(), index.end()));
	int idx = std::distance(index.begin(), index.end());
}

int TriangleMesh::getMiddlePoint(int p1, int p2)
{
	// calculate it
	vec3f point1 = temp_vertex[p1];
	vec3f point2 = temp_vertex[p2];
	vec3f middle = (point1 + point2) / 2.0f;

	// add vertex
	float norm = sqrt(dot(middle, middle));
	middle = middle / norm;
	temp_vertex.push_back(middle);

	//return index of the vertex
	return temp_vertex.size() - 1;
}

/*! constructor - performs all setup, including initializing
		optix, creates module, pipeline, programs, SBT, etc. */
OptixSetup::OptixSetup(const std::vector<TriangleMesh> &meshes)
	: meshes(meshes)
{
	initOptix();

	std::cout << "#osc: creating optix context ..." << std::endl;
	createContext();

	std::cout << "#osc: setting up module ..." << std::endl;
	createModule();

	std::cout << "#osc: creating raygen programs ..." << std::endl;
	createRaygenPrograms();
	std::cout << "#osc: creating miss programs ..." << std::endl;
	createMissPrograms();
	std::cout << "#osc: creating hitgroup programs ..." << std::endl;
	createHitgroupPrograms();

	std::cout << "#osc: building acceleration structure ..." << std::endl;
	SoundItem::traversable = buildAccel();
	// launchParams.traversable = buildAccel();
	// launchParams.time_bins = 2500;
	// launchParams.freq_bands = 9;
	// //launchParams.hist_res = 192;
	// unsigned long long STRIDE = launchParams.time_bins * launchParams.freq_bands;
	// unsigned long long STRIDE_POW_2 = next_pow_2(STRIDE);
	// unsigned long long MAX_MICS = 10;
	// unsigned long long N_RAYS = 1024;
	// checkCudaErrors(cudaMalloc(&(launchParams.d_histogram), STRIDE * MAX_MICS * sizeof(float)));
	// checkCudaErrors(cudaMalloc(&(launchParams.d_transmitted), launchParams.freq_bands * N_RAYS * sizeof(float)));
	// kernels::fillWithZeroesKernel(launchParams.d_histogram, STRIDE * MAX_MICS);
	// kernels::fillWithZeroesKernel(launchParams.d_transmitted, launchParams.freq_bands * MAX_MICS);
	// DEBUG_CHECK();
	std::cout << "#osc: setting up optix pipeline ..." << std::endl;
	createPipeline();
	SoundItem::pipeline = pipeline;

	std::cout << "#osc: building SBT ..." << std::endl;
	buildSBT();

	SoundItem::sbt = sbt;

	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

	std::cout << GDT_TERMINAL_GREEN;
	std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
	std::cout << GDT_TERMINAL_DEFAULT;

	std::cout << "Launching dummy kernel" << std::endl;
	launchParams.frame.size.x = 1;

	launchParams.dummy_launch = true;
	render();
	launchParams.dummy_launch = false;
}

OptixTraversableHandle OptixSetup::buildAccel()
{
	// meshes.resize(1);

	vertexBuffer.resize(meshes.size());
	indexBuffer.resize(meshes.size());

	OptixTraversableHandle asHandle{0};

	// ==================================================================
	// triangle inputs
	// ==================================================================
	std::vector<OptixBuildInput> triangleInput(meshes.size());
	std::vector<CUdeviceptr> d_vertices(meshes.size());
	std::vector<CUdeviceptr> d_indices(meshes.size());
	std::vector<uint32_t> triangleInputFlags(meshes.size());
	DEBUG_CHECK();
	for (int meshID = 0; meshID < meshes.size(); meshID++)
	{
		// upload the model to the device: the builder
		TriangleMesh &model = meshes[meshID];
		vertexBuffer[meshID].alloc_and_upload(model.vertex);
		indexBuffer[meshID].alloc_and_upload(model.index);

		triangleInput[meshID] = {};
		triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
		d_indices[meshID] = indexBuffer[meshID].d_pointer();

		triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		triangleInput[meshID].triangleArray.numVertices = (int)model.vertex.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

		triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInput[meshID].triangleArray.numIndexTriplets = (int)model.index.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

		triangleInputFlags[meshID] = 0;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
	// ==================================================================
	// BLAS setup
	// ==================================================================

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
											 &accelOptions,
											 triangleInput.data(),
											 (int)meshes.size(), // num_build_inputs
											 &blasBufferSizes));

	// ==================================================================
	// prepare compaction
	// ==================================================================

	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(optixContext,
								/* stream */ 0,
								&accelOptions,
								triangleInput.data(),
								(int)meshes.size(),
								tempBuffer.d_pointer(),
								tempBuffer.sizeInBytes,

								outputBuffer.d_pointer(),
								outputBuffer.sizeInBytes,

								&asHandle,

								&emitDesc, 1));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// perform compaction
	// ==================================================================
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext,
								  /*stream:*/ 0,
								  asHandle,
								  asBuffer.d_pointer(),
								  asBuffer.sizeInBytes,
								  &asHandle));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// aaaaaand .... clean up
	// ==================================================================
	outputBuffer.free(); // << the UNcompacted, temporary output buffer
	tempBuffer.free();
	compactedSizeBuffer.free();

	return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void OptixSetup::initOptix()
{
	std::cout << "#osc: initializing optix..." << std::endl;

	// -------------------------------------------------------
	// check for available optix7 capable devices
	// -------------------------------------------------------
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("#osc: no CUDA capable devices found!");
	std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

	// -------------------------------------------------------
	// initialize optix
	// -------------------------------------------------------
	OPTIX_CHECK(optixInit());
	std::cout << GDT_TERMINAL_GREEN
			  << "#osc: successfully initialized optix... yay!"
			  << GDT_TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level,
						   const char *tag,
						   const char *message,
						   void *)
{
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
		example, only for the primary GPU device) */
void OptixSetup::createContext()
{
	// for this sample, do everything on one device
	const int deviceID = 0;
	CUDA_CHECK(SetDevice(deviceID));
	CUDA_CHECK(StreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
		to use. in this simple example, we use a single module from a
		single .cu file, using a single embedded ptx string */
void OptixSetup::createModule()
{
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	//pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.overrideUsesMotionBlur = false;
	pipelineLinkOptions.maxTraceDepth = 31;

	const std::string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
										 &moduleCompileOptions,
										 &pipelineCompileOptions,
										 ptxCode.c_str(),
										 ptxCode.size(),
										 log, &sizeof_log,
										 &module));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the raygen program(s) we are going to use */
void OptixSetup::createRaygenPrograms()
{
	// we do a single ray gen program in this example:
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&raygenPGs[0]));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void OptixSetup::createMissPrograms()
{
	// we do a single ray gen program in this example:
	missPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__radiance";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&missPGs[0]));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void OptixSetup::createHitgroupPrograms()
{
	// for this simple example, we set up a single hit group
	hitgroupPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&hitgroupPGs[0]));
	DEBUG_CHECK();
	if (sizeof_log > 1)
		PRINT(log);
	OptixStackSizes stackSizes;
	optixProgramGroupGetStackSize(hitgroupPGs[0], &stackSizes);
}

/*! assembles the full pipeline of all programs */
void OptixSetup::createPipeline()
{
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optixContext,
									&pipelineCompileOptions,
									&pipelineLinkOptions,
									programGroups.data(),
									(int)programGroups.size(),
									log, &sizeof_log,
									&pipeline));
	if (sizeof_log > 1)
		PRINT(log);

	OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
										  pipeline,
										  /* [in] The direct stack size requirement for direct
				   callables invoked from IS or AH. */
										  2 * 1024,
										  /* [in] The direct stack size requirement for direct
				   callables invoked from RG, MS, or CH.  */
										  2 * 1024,
										  /* [in] The continuation stack requirement. */
										  2 * 1024,
										  /* [in] The maximum depth of a traversable graph
				   passed to trace. */
										  1));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! constructs the shader binding table */
void OptixSetup::buildSBT()
{
	// ------------------------------------------------------------------
	// build raygen records
	// ------------------------------------------------------------------
	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++)
	{
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++)
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordsBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------
	int numObjects = (int)meshes.size();
	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshID = 0; meshID < numObjects; meshID++)
	{
		HitgroupRecord rec;
		// all meshes use the same code, so all same hit group
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
		rec.data.color = meshes[meshID].color;
		rec.data.vertex = (vec3f *)vertexBuffer[meshID].d_pointer();
		rec.data.index = (vec3i *)indexBuffer[meshID].d_pointer();
		rec.data.isMic = meshID == 0 ? false : true;
		float pyroom_absorption = 0.3; // TODO: Currently deprecated in pyroomacoustics. Need to change
		rec.data.absorption = 1. - (1. - pyroom_absorption) * (1. - pyroom_absorption);
		rec.data.micID = meshID == 0 ? -1 : 0;
		rec.data.pos = meshes[meshID].m_center;

		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/*! render one frame */
void OptixSetup::render()
{
	// sanity check: make sure we launch only after first resize is
	// already done:
	if (launchParams.frame.size.x == 0)
		return;
	//for (int i = 1; i < 256; i++) {
	//printf("i: %i\n", i);
	//launchParams.frame.size.x = 1025;
	//launchParams.frame.size.y = 2;
	launchParamsBuffer.upload(&launchParams, 1);

	OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
							pipeline, stream,
							/*! parameters and SBT */
							launchParamsBuffer.d_pointer(),
							launchParamsBuffer.sizeInBytes,
							&sbt,
							/*! dimensions of the launch: */
							launchParams.frame.size.x,
							1, // launchParams.frame.size.y,
							1));
	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)
	CUDA_SYNC_CHECK();
	//}
}

/*! set camera to render with */
void OptixSetup::setCamera(const Camera &camera)
{
	lastSetCamera = camera;
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = normalize(camera.at - camera.from);
	const float cosFovy = 0.66f;
	const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
	launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
	launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal,
															 launchParams.camera.direction));
}

/*! resize frame buffer to given resolution */
void OptixSetup::resize(const vec2i &newSize)
{
	// resize our cuda frame buffer
	colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

	// update the launch parameters that we'll pass to the optix
	// launch:
	launchParams.frame.size = newSize;
	launchParams.frame.colorBuffer = (uint32_t *)colorBuffer.d_pointer();

	// and re-set the camera, since aspect may have changed
	setCamera(lastSetCamera);
}

/*! download the rendered color buffer */
void OptixSetup::downloadPixels(uint32_t h_pixels[])
{
	colorBuffer.download(h_pixels,
						 launchParams.frame.size.x * launchParams.frame.size.y);
}

void OptixSetup::add_mic(Microphone *mic)
{
	m_mics.push_back(mic);
	for (int i = 0; i < m_sources.size(); i++) {
		m_sources[i]->add_mic(*mic);
	}
}
void OptixSetup::add_source(SoundSource *src)
{
	m_sources.push_back(src);
}
std::vector<SoundSource*> OptixSetup::get_sources(){
	return m_sources;
}
std::vector<Microphone*> OptixSetup::get_microphones(){
	return m_mics;
}
void OptixSetup::auralize()
{
	for (int i = 0; i < m_sources.size(); i++)
	{
		m_sources[i]->trace();
		//m_sources[i]->compute_IRs();
		
	}
	//m_sources[0]->convolve_file("../../Ex_441_Mono.wav", "output.wav", 0);
	/*for (int i = 0; i < m_mics.size(); i++)
	{
		m_mics[i]->zero_output();
	}*/
}

} // namespace osc
