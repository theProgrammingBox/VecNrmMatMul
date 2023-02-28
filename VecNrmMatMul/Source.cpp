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
	constexpr float TWOF = 2.0f;
	constexpr float NONEF = -1.0f;
	
	constexpr float LEARNING_RATE = 0.01f;
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

class Visualizer : public olc::PixelGameEngine
{
public:
	uint32_t vecDim = 2;
	uint32_t numInputVecs = 5;
	uint32_t numTargetVecs = 5;
	float* inputVec;
	float* targetVec;
	float* productMatrix;
	float* squaredMagnitudeMatrix1;
	float* squaredMagnitudeMatrix2;
	float* magProductMatrix;
	float* invSqrtProductMatrix;
	float* dotProductMatrix;

	float* targetSimilairtyMatrix;
	float* dotProductDerivitiveMatrix;
	float* productDerivitiveMatrix;
	float* inputVecDerivitiveMatrix;
	float* targetVecDerivitiveMatrix;
	float* invSqrtProductDerivitiveMatrix;
	float* magProductDerivitiveMatrix;
	float* squaredMagnitudeMatrix1Derivitive;
	float* squaredMagnitudeMatrix2Derivitive;

	float orgin[2];
	
	Visualizer()
	{
		sAppName = "Visualizing Vector Norm Gradient";
	}

public:
	bool OnUserCreate() override
	{
		inputVec = new float[vecDim * numInputVecs];
		targetVec = new float[vecDim * numTargetVecs];
		productMatrix = new float[numInputVecs * numTargetVecs];
		squaredMagnitudeMatrix1 = new float[numInputVecs];
		squaredMagnitudeMatrix2 = new float[numTargetVecs];
		magProductMatrix = new float[numInputVecs * numTargetVecs];
		invSqrtProductMatrix = new float[numInputVecs * numTargetVecs];
		dotProductMatrix = new float[numInputVecs * numTargetVecs];

		targetSimilairtyMatrix = new float[numInputVecs * numTargetVecs];
		dotProductDerivitiveMatrix = new float[numInputVecs * numTargetVecs];
		productDerivitiveMatrix = new float[numInputVecs * numTargetVecs];
		inputVecDerivitiveMatrix = new float[vecDim * numInputVecs];
		targetVecDerivitiveMatrix = new float[vecDim * numTargetVecs];
		invSqrtProductDerivitiveMatrix = new float[numInputVecs * numTargetVecs];
		magProductDerivitiveMatrix = new float[numInputVecs * numTargetVecs];
		squaredMagnitudeMatrix1Derivitive = new float[numInputVecs];
		squaredMagnitudeMatrix2Derivitive = new float[numTargetVecs];

		cpuGenerateUniform(inputVec, vecDim * numInputVecs, -1, 1);
		cpuGenerateUniform(targetVec, vecDim * numTargetVecs, -1, 1);
		cpuGenerateUniform(targetSimilairtyMatrix, numInputVecs * numTargetVecs, -1, 1);

		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetMouse(0).bHeld)
		{
			targetVec[0] = (GetMouseX() - orgin[0]) * 0.01f;
			targetVec[1] = (GetMouseY() - orgin[1]) * 0.01f;
		}
		
		Clear(olc::BLACK);
		for (uint32_t i = numTargetVecs; i--;)
			DrawLine(orgin[0], orgin[1], orgin[0] + targetVec[i * vecDim] * 100.0f, orgin[1] + targetVec[i * vecDim + 1] * 100.0f, olc::RED);
		for (uint32_t i = numInputVecs; i--;)
			DrawLine(orgin[0], orgin[1], orgin[0] + inputVec[i * vecDim] * 100.0f, orgin[1] + inputVec[i * vecDim + 1] * 100.0f, olc::GREEN);
		
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

		memcpy(dotProductDerivitiveMatrix, targetSimilairtyMatrix, sizeof(float) * numInputVecs * numTargetVecs);
		cpuSaxpy(numInputVecs * numTargetVecs, &GLOBAL::NONEF, dotProductMatrix, 1, dotProductDerivitiveMatrix, 1);

		cpuMultiply(dotProductDerivitiveMatrix, invSqrtProductMatrix, productDerivitiveMatrix, numInputVecs * numTargetVecs);

		cpuSgemmStridedBatched(
			false, false,
			vecDim, numInputVecs, numTargetVecs,
			&GLOBAL::ONEF,
			targetVec, vecDim, 0,
			productDerivitiveMatrix, numTargetVecs, 0,
			&GLOBAL::ZEROF,
			inputVecDerivitiveMatrix, vecDim, 0,
			1);

		cpuSgemmStridedBatched(
			false, true,
			vecDim, numTargetVecs, numInputVecs,
			&GLOBAL::ONEF,
			inputVec, vecDim, 0,
			productDerivitiveMatrix, numTargetVecs, 0,
			&GLOBAL::ZEROF,
			targetVecDerivitiveMatrix, vecDim, 0,
			1);

		cpuMultiply(dotProductDerivitiveMatrix, productMatrix, invSqrtProductDerivitiveMatrix, numInputVecs * numTargetVecs);

		invSqrtDerivitive(invSqrtProductMatrix, invSqrtProductDerivitiveMatrix, magProductDerivitiveMatrix, numInputVecs * numTargetVecs);
		
		cpuSgemmStridedBatched(
			true, false,
			numInputVecs, 1, numTargetVecs,
			&GLOBAL::ONEF,
			magProductDerivitiveMatrix, numTargetVecs, 0,
			squaredMagnitudeMatrix2, numTargetVecs, 0,
			&GLOBAL::ZEROF,
			squaredMagnitudeMatrix1Derivitive, numInputVecs, 0,
			1);

		cpuSgemmStridedBatched(
			false, false,
			numTargetVecs, 1, numInputVecs,
			&GLOBAL::ONEF,
			magProductDerivitiveMatrix, numTargetVecs, 0,
			squaredMagnitudeMatrix1, numInputVecs, 0,
			&GLOBAL::ZEROF,
			squaredMagnitudeMatrix2Derivitive, numTargetVecs, 0,
			1);
		
		cpuSgemmStridedBatched(
			false, false,
			vecDim, 1, 1,
			&GLOBAL::TWOF,
			inputVec, vecDim, vecDim,
			squaredMagnitudeMatrix1Derivitive, numInputVecs, 1,
			&GLOBAL::ONEF,
			inputVecDerivitiveMatrix, vecDim, vecDim,
			numInputVecs);

		cpuSgemmStridedBatched(
			false, false,
			vecDim, 1, 1,
			&GLOBAL::TWOF,
			targetVec, vecDim, vecDim,
			squaredMagnitudeMatrix2Derivitive, numTargetVecs, 1,
			&GLOBAL::ONEF,
			targetVecDerivitiveMatrix, vecDim, vecDim,
			numTargetVecs);

		cpuSaxpy(numInputVecs * vecDim, &GLOBAL::LEARNING_RATE, inputVecDerivitiveMatrix, 1, inputVec, 1);
		cpuSaxpy(numTargetVecs * vecDim, &GLOBAL::LEARNING_RATE, targetVecDerivitiveMatrix, 1, targetVec, 1);

		if (GetKey(olc::Key::SPACE).bPressed)
		{
			/*PrintMatrix(inputVec, numInputVecs, vecDim, "inputVec");
			PrintMatrix(targetVec, numTargetVecs, vecDim, "targetVec");
			PrintMatrix(productMatrix, numInputVecs, numTargetVecs, "productMatrix");*/
			PrintMatrix(squaredMagnitudeMatrix1, 1, numInputVecs, "squaredMagnitudeMatrix1");
			PrintMatrix(squaredMagnitudeMatrix2, 1, numTargetVecs, "squaredMagnitudeMatrix2");
			/*PrintMatrix(magProductMatrix, numInputVecs, numTargetVecs, "magProductMatrix");
			PrintMatrix(invSqrtProductMatrix, numInputVecs, numTargetVecs, "invSqrtProductMatrix");
			PrintMatrix(dotProductMatrix, numInputVecs, numTargetVecs, "dotProductMatrix");
			PrintMatrix(targetSimilairtyMatrix, numInputVecs, numTargetVecs, "targetSimilairtyMatrix");
			PrintMatrix(dotProductDerivitiveMatrix, numInputVecs, numTargetVecs, "dotProductDerivitiveMatrix");
			PrintMatrix(productDerivitiveMatrix, numInputVecs, numTargetVecs, "productDerivitiveMatrix");
			PrintMatrix(inputVecDerivitiveMatrix, numInputVecs, vecDim, "inputVecDerivitiveMatrix");
			PrintMatrix(targetVecDerivitiveMatrix, numTargetVecs, vecDim, "targetVecDerivitiveMatrix");
			PrintMatrix(invSqrtProductDerivitiveMatrix, numInputVecs, numTargetVecs, "invSqrtProductDerivitiveMatrix");*/
			PrintMatrix(magProductDerivitiveMatrix, numInputVecs, numTargetVecs, "magProductDerivitiveMatrix");
			/*PrintMatrix(squaredMagnitudeMatrix1Derivitive, 1, numInputVecs, "squaredMagnitudeMatrix1Derivitive");
			PrintMatrix(squaredMagnitudeMatrix2Derivitive, 1, numTargetVecs, "squaredMagnitudeMatrix2Derivitive");
			PrintMatrix(inputVecDerivitiveMatrix, numInputVecs, vecDim, "inputVecDerivitiveMatrix");
			PrintMatrix(targetVecDerivitiveMatrix, numTargetVecs, vecDim, "targetVecDerivitiveMatrix");*/
		}
		
		return true;
	}
};

int main()
{
	Visualizer visualizer;
	if (visualizer.Construct(1600, 900, 1, 1))
		visualizer.Start();
	
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