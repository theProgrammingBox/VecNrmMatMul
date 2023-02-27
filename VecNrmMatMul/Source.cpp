#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;

/*
IMPORTANT LESSONS
1. With Euler, the length of the vector increases very noticeably.
2. With Runge Kutta 4, the length of the vector remains a lot more stable
3. For a learning rate of 0.1, the length of the vector increases by about 0.0001 per frame with runge kutta 4
4. If the initial vector length is very small, with the same learning rate, the length of the vector remains incredibly stable
5. Error is larger when the initial vector length is large
*/

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, 4, seed);
		state[1] = Hash((uint8_t*)&seed, 4, state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, 4, nanosecond());
		result = Hash((uint8_t*)&result, 4, microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, 4, seed);
		state[1] = Hash((uint8_t*)&seed, 4, state[0]);
	}

	uint32_t Ruint32()	// XORSHIFT128+
	{
		uint64_t a = state[0];
		uint64_t b = state[1];
		state[0] = b;
		a ^= a << 23;
		state[1] = a ^ b ^ (a >> 18) ^ (b >> 5);
		return uint32_t((state[1] + b) >> 16);
	}

	float Rfloat(float min = 0, float max = 1) { return min + (max - min) * Ruint32() * 2.3283064371e-10; }

	static uint32_t Hash(const uint8_t* key, size_t len, uint32_t seed = 0)	// MurmurHash3
	{
		uint32_t h = seed;
		uint32_t k;
		for (size_t i = len >> 2; i; i--) {
			memcpy(&k, key, 4);
			key += 4;
			h ^= murmur_32_scramble(k);
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}
		k = 0;
		for (size_t i = len & 3; i; i--) {
			k <<= 8;
			k |= key[i - 1];
		}
		h ^= murmur_32_scramble(k);
		h ^= len;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

private:
	uint64_t state[2];

	static uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	static uint32_t nanosecond() { return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
	static uint32_t microsecond() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
};

namespace GLOBAL
{
	Random random(Random::MakeSeed(0));
	constexpr float ZEROF = 0.0f;
	constexpr float ONEF = 1.0f;
	
	constexpr float LEARNING_RATE = 0.1f;
	constexpr float HALF_LEARNING_RATE = LEARNING_RATE * 0.5f;
	constexpr float THIRD_LEARNING_RATE = LEARNING_RATE * 0.33333333333333333333333333333333f;
	constexpr float SIXTH_LEARNING_RATE = LEARNING_RATE * 0.16666666666666666666666666666667f;

	constexpr float applied[4] = { 0.0f, HALF_LEARNING_RATE, HALF_LEARNING_RATE, LEARNING_RATE };
	constexpr float summed[4] = { SIXTH_LEARNING_RATE, THIRD_LEARNING_RATE, THIRD_LEARNING_RATE, SIXTH_LEARNING_RATE };
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::random.Rfloat(min, max);
}

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuSaxpy(int N, const float* alpha, const float* X, int incX, float* Y, int incY)
{
	for (int i = N; i--;)
		Y[i * incY] += *alpha * X[i * incX];
}

float invSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

float cpuNormDot(uint32_t size, float* vec1, float* vec2, float* vec1Gradient, float* vec2Gradient) {
	float sum1[1];
	float sum2[1];
	float dot[1];

	cpuSgemmStridedBatched(
		false, false,
		1, 1, size,
		&GLOBAL::ONEF,
		vec1, 1, 0,
		vec1, size, 0,
		&GLOBAL::ZEROF,
		sum1, 1, 0,
		1);

	cpuSgemmStridedBatched(
		false, false,
		1, 1, size,
		&GLOBAL::ONEF,
		vec2, 1, 0,
		vec2, size, 0,
		&GLOBAL::ZEROF,
		sum2, 1, 0,
		1);

	cpuSgemmStridedBatched(
		false, false,
		1, 1, size,
		&GLOBAL::ONEF,
		vec1, 1, 0,
		vec2, size, 0,
		&GLOBAL::ZEROF,
		dot, 1, 0,
		1);
	
	float invMagsProduct = invSqrt(*sum1 * *sum2);
	
	for (uint32_t j = size; j--;)
		vec1Gradient[j] = (vec2[j] * (*sum1 - vec1[j] * vec1[j]) + vec1[j] * (vec1[j] * vec2[j] - *dot)) * invMagsProduct;
	
	for (uint32_t j = size; j--;)
		vec2Gradient[j] = (vec1[j] * (*sum2 - vec2[j] * vec2[j]) + vec2[j] * (vec2[j] * vec1[j] - *dot)) * invMagsProduct;

	return *dot * invMagsProduct;
}

class Visualizer : public olc::PixelGameEngine
{
public:
	float orgin[2];
	float vec[2];
	float mouseVec[2];
	float tempVec[2];
	float savedVec[2];
	float vecDerivitive[2];
	float mouseVecDerivitive[2];
	uint32_t rungeKuttaStep;
	
	Visualizer()
	{
		sAppName = "Visualizing Vector Norm Gradient";
	}

public:
	bool OnUserCreate() override
	{
		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;

		vec[0] = 1000;
		vec[1] = 0;

		mouseVec[0] = -100;
		mouseVec[1] = 0;

		rungeKuttaStep = 0;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetMouse(0).bHeld)
		{
			mouseVec[0] = GetMouseX() - orgin[0];
			mouseVec[1] = GetMouseY() - orgin[1];
		}
		
		Clear(olc::BLACK);

		DrawLine(orgin[0], orgin[1], orgin[0] + vec[0] * 0.1, orgin[1] + vec[1] * 0.1, olc::RED);
		DrawLine(orgin[0], orgin[1], orgin[0] + mouseVec[0], orgin[1] + mouseVec[1], olc::GREEN);

		if (rungeKuttaStep == 0)
			memcpy(savedVec, vec, sizeof(float) * 2);
		memcpy(tempVec, savedVec, sizeof(float) * 2);
		//printf("vecDerivitive: %f, %f\n", vecDerivitive[0], vecDerivitive[1]);
		if (rungeKuttaStep != 0)
			cpuSaxpy(2, &GLOBAL::applied[rungeKuttaStep], vecDerivitive, 1, tempVec, 1);
		
		cpuNormDot(2, tempVec, mouseVec, vecDerivitive, mouseVecDerivitive);
		
		cpuSaxpy(2, &GLOBAL::summed[rungeKuttaStep], vecDerivitive, 1, vec, 1);
		rungeKuttaStep *= ++rungeKuttaStep != 4;
		
		float vecMag = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
		DrawString(10, 10, "vec magnitude: " + std::to_string(vecMag), olc::WHITE, 1);

		return true;
	}
};

void PrintMatrix(float* arr, uint32_t rows, uint32_t cols, const char* label) {
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

void cpuInvSqrt(float* input, float* output, uint32_t size)
{
	for (uint32_t i = size; i--;)
		output[i] = invSqrt(input[i]);
}

void invSqrtDerivitive(float* invSqrtResult, float* gradient, float* output, uint32_t size)
{
	for (uint32_t i = size; i--;)
		output[i] = -0.5f * gradient[i] * invSqrtResult[i] * invSqrtResult[i] * invSqrtResult[i];
}

void cpuMultiply(float* input, float* input2, float* output, uint32_t size)
{
	for (uint32_t i = size; i--;)
		output[i] = input[i] * input2[i];
}

int main()
{
	/*Visualizer visualizer;
	if (visualizer.Construct(1600, 900, 1, 1))
		visualizer.Start();*/

	GLOBAL::random.Seed(0);
	
	uint32_t vecDim = 2;
	uint32_t numInputVecs = 2;
	uint32_t numTargetVecs = 2;
	float* inputVec;
	float* targetVec;
	float* productMatrix;
	float* squaredMagnitudeMatrix1;
	float* squaredMagnitudeMatrix2;
	float* magProductMatrix;
	float* invSqrtProductMatrix;
	float* dotProductMatrix;
	
	float* vecDerivitive;
	float* targetVecDerivitive;

	inputVec = new float[vecDim * numInputVecs];
	targetVec = new float[vecDim * numTargetVecs];
	productMatrix = new float[numInputVecs * numTargetVecs];
	squaredMagnitudeMatrix1 = new float[numInputVecs];
	squaredMagnitudeMatrix2 = new float[numTargetVecs];
	magProductMatrix = new float[numInputVecs * numTargetVecs];
	invSqrtProductMatrix = new float[numInputVecs * numTargetVecs];
	dotProductMatrix = new float[numInputVecs * numTargetVecs];
	
	vecDerivitive = new float[vecDim * numInputVecs];
	targetVecDerivitive = new float[vecDim * numTargetVecs];

	cpuGenerateUniform(inputVec, vecDim * numInputVecs, -1, 1);
	cpuGenerateUniform(targetVec, vecDim * numTargetVecs, -1, 1);
	
	cpuSgemmStridedBatched(
		true, false,
		numInputVecs, numTargetVecs, vecDim,
		&GLOBAL::ONEF,
		targetVec, vecDim, 0,
		inputVec, vecDim, 0,
		&GLOBAL::ZEROF,
		productMatrix, numTargetVecs, 0,
		1);

	cpuSgemmStridedBatched(
		true, false,
		1, 1, vecDim,
		&GLOBAL::ONEF,
		inputVec, vecDim, vecDim,
		inputVec, vecDim, vecDim,
		&GLOBAL::ZEROF,
		squaredMagnitudeMatrix1, numInputVecs, 1,
		numInputVecs);

	cpuSgemmStridedBatched(
		true, false,
		1, 1, vecDim,
		&GLOBAL::ONEF,
		targetVec, vecDim, vecDim,
		targetVec, vecDim, vecDim,
		&GLOBAL::ZEROF,
		squaredMagnitudeMatrix2, numTargetVecs, 1,
		numTargetVecs);

	cpuSgemmStridedBatched(
		false, true,
		numTargetVecs, numInputVecs, 1,
		&GLOBAL::ONEF,
		squaredMagnitudeMatrix2, 1, 0,
		squaredMagnitudeMatrix1, 1, 0,
		&GLOBAL::ZEROF,
		magProductMatrix, numInputVecs, 0,
		1);

	cpuInvSqrt(magProductMatrix, invSqrtProductMatrix, numInputVecs * numTargetVecs);

	cpuMultiply(productMatrix, invSqrtProductMatrix, dotProductMatrix, numInputVecs * numTargetVecs);
	
	PrintMatrix(inputVec, numInputVecs, vecDim, "inputVec");
	PrintMatrix(targetVec, numTargetVecs, vecDim, "targetVec");
	PrintMatrix(productMatrix, numInputVecs, numTargetVecs, "productMatrix");
	PrintMatrix(squaredMagnitudeMatrix1, 1, numInputVecs, "squaredMagnitudeMatrix1");
	PrintMatrix(squaredMagnitudeMatrix2, 1, numTargetVecs, "squaredMagnitudeMatrix2");
	PrintMatrix(magProductMatrix, numInputVecs, numTargetVecs, "magProductMatrix");
	PrintMatrix(invSqrtProductMatrix, numInputVecs, numTargetVecs, "invSqrtProductMatrix");
	PrintMatrix(dotProductMatrix, numInputVecs, numTargetVecs, "dotProductMatrix");
	/*
	p_{ 11 } = v_{ 111 }\cdot v_{ 211 } + v_{ 112 }\cdot v_{ 212 }
	p_{ 12 } = v_{ 111 }\cdot v_{ 221 } + v_{ 112 }\cdot v_{ 222 }
	p_{ 21 } = v_{ 121 }\cdot v_{ 211 } + v_{ 122 }\cdot v_{ 212 }
	p_{ 22 } = v_{ 121 }\cdot v_{ 221 } + v_{ 122 }\cdot v_{ 222 }

	s_{11} = v_{111}^2 + v_{112}^2
	s_{12} = v_{121}^2 + v_{122}^2
	
	s_{21} = v_{211}^2 + v_{212}^2
	s_{22} = v_{221}^2 + v_{222}^2

	m_{11} = s_{11} * s_{21}
	m_{12} = s_{11} * s_{22}
	m_{21} = s_{12} * s_{21}
	m_{22} = s_{12} * s_{22}

	i_{11}=\frac{1}{\sqrt{m_{11}}}
	i_{12}=\frac{1}{\sqrt{m_{12}}}
	i_{21}=\frac{1}{\sqrt{m_{21}}}
	i_{22}=\frac{1}{\sqrt{m_{22}}}

	D_{11}=p_{11} * i_{11}
	D_{12}=p_{12} * i_{12}
	D_{21}=p_{21} * i_{21}
	D_{22}=p_{22} * i_{22}
	*/
	
	return 0;
}