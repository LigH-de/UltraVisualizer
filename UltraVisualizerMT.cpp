#include <windows.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <complex>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "avisynth.h"

#if !defined(_M_ARM64EC) && (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__))
#define ULTRAVIS_X86
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef HAVE_STD_CLAMP
template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
	return (v < lo) ? lo : (hi < v) ? hi : v;
}
#endif

// Structure for an RGBA pixel of the internal clip
struct PixelRGBA
{
	BYTE r, g, b, a;
};

// Process complete SIMD blocks in [2, end) and return the start of the scalar tail.
using DecayInterior = int (*)(const PixelRGBA* row, const PixelRGBA* up1, const PixelRGBA* dn1,
	const PixelRGBA* up2, const PixelRGBA* dn2, PixelRGBA* out, int end);
using OverlayRow = void (*)(BYTE* dst, const PixelRGBA* overlay, int width);

static void OverlayScalar(BYTE* dst, const PixelRGBA* overlay, int width)
{
	for (; width > 0; --width, dst += 3, ++overlay)
	{
		const int alpha = overlay->a;
		if (alpha == 0)
			continue;
		const int inv = 255 - alpha;
		// AviSynth RGB24 stores the three channels in BGR order.
		dst[0] = static_cast<BYTE>((dst[0] * inv + overlay->b * alpha) / 255);
		dst[1] = static_cast<BYTE>((dst[1] * inv + overlay->g * alpha) / 255);
		dst[2] = static_cast<BYTE>((dst[2] * inv + overlay->r * alpha) / 255);
	}
}

#ifdef ULTRAVIS_X86
static_assert(sizeof(PixelRGBA) == 4, "SIMD kernels require packed RGBA");

// Keep the rest of this translation unit at the target baseline (MSVC Win32: /arch:IA32).
// MSVC lowers explicit intrinsics without a global /arch:AVX2 switch.
#if defined(__clang__) || defined(__GNUC__)
#define UV_TARGET(isa) __attribute__((target(isa)))
#define UV_KERNEL(isa) UV_TARGET(isa) __attribute__((noinline))
#else
#define UV_TARGET(isa)
#define UV_KERNEL(isa) __declspec(noinline)
#endif

// Each sum is at most 4 * 255. For this range mulhi(sum, ceil(65536 / d))
// is exact integer division by d (RGB: 8, near alpha: 10, far alpha: 12).
static UV_KERNEL("sse2") int DecaySse2(const PixelRGBA* row, const PixelRGBA* up1, const PixelRGBA* dn1,
	const PixelRGBA* up2, const PixelRGBA* dn2, PixelRGBA* out, int end)
{
	auto sum4 = [](const PixelRGBA* a, const PixelRGBA* b, const PixelRGBA* c, const PixelRGBA* d,
		__m128i& lo, __m128i& hi) UV_TARGET("sse2") {
		const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
		const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
		const __m128i vc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c));
		const __m128i vd = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d));
		const __m128i zero = _mm_setzero_si128();
		lo = _mm_add_epi16(_mm_add_epi16(_mm_unpacklo_epi8(va, zero), _mm_unpacklo_epi8(vb, zero)),
			_mm_add_epi16(_mm_unpacklo_epi8(vc, zero), _mm_unpacklo_epi8(vd, zero)));
		hi = _mm_add_epi16(_mm_add_epi16(_mm_unpackhi_epi8(va, zero), _mm_unpackhi_epi8(vb, zero)),
			_mm_add_epi16(_mm_unpackhi_epi8(vc, zero), _mm_unpackhi_epi8(vd, zero)));
	};
	const __m128i near_scale = _mm_set_epi16(6554, 8192, 8192, 8192, 6554, 8192, 8192, 8192);
	const __m128i far_scale = _mm_set_epi16(5462, 8192, 8192, 8192, 5462, 8192, 8192, 8192);
	int x = 2;
	for (; x <= end - 4; x += 4)
	{
		__m128i near_lo, near_hi, far_lo, far_hi;
		sum4(up1 + x - 1, up1 + x + 1, dn1 + x - 1, dn1 + x + 1, near_lo, near_hi);
		sum4(up2 + x, dn2 + x, row + x - 2, row + x + 2, far_lo, far_hi);
		const __m128i lo = _mm_add_epi16(_mm_mulhi_epu16(near_lo, near_scale), _mm_mulhi_epu16(far_lo, far_scale));
		const __m128i hi = _mm_add_epi16(_mm_mulhi_epu16(near_hi, near_scale), _mm_mulhi_epu16(far_hi, far_scale));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), _mm_packus_epi16(lo, hi));
	}
	return x;
}

static UV_KERNEL("avx2") int DecayAvx2(const PixelRGBA* row, const PixelRGBA* up1, const PixelRGBA* dn1,
	const PixelRGBA* up2, const PixelRGBA* dn2, PixelRGBA* out, int end)
{
	auto sum4 = [](const PixelRGBA* a, const PixelRGBA* b, const PixelRGBA* c, const PixelRGBA* d,
		__m256i& lo, __m256i& hi) UV_TARGET("avx2") {
		const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
		const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
		const __m256i vc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c));
		const __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d));
		const __m256i zero = _mm256_setzero_si256();
		lo = _mm256_add_epi16(_mm256_add_epi16(_mm256_unpacklo_epi8(va, zero), _mm256_unpacklo_epi8(vb, zero)),
			_mm256_add_epi16(_mm256_unpacklo_epi8(vc, zero), _mm256_unpacklo_epi8(vd, zero)));
		hi = _mm256_add_epi16(_mm256_add_epi16(_mm256_unpackhi_epi8(va, zero), _mm256_unpackhi_epi8(vb, zero)),
			_mm256_add_epi16(_mm256_unpackhi_epi8(vc, zero), _mm256_unpackhi_epi8(vd, zero)));
	};
	const __m256i near_scale = _mm256_broadcastsi128_si256(_mm_set_epi16(6554, 8192, 8192, 8192, 6554, 8192, 8192, 8192));
	const __m256i far_scale = _mm256_broadcastsi128_si256(_mm_set_epi16(5462, 8192, 8192, 8192, 5462, 8192, 8192, 8192));
	int x = 2;
	for (; x <= end - 8; x += 8)
	{
		__m256i near_lo, near_hi, far_lo, far_hi;
		sum4(up1 + x - 1, up1 + x + 1, dn1 + x - 1, dn1 + x + 1, near_lo, near_hi);
		sum4(up2 + x, dn2 + x, row + x - 2, row + x + 2, far_lo, far_hi);
		const __m256i lo = _mm256_add_epi16(_mm256_mulhi_epu16(near_lo, near_scale), _mm256_mulhi_epu16(far_lo, far_scale));
		const __m256i hi = _mm256_add_epi16(_mm256_mulhi_epu16(near_hi, near_scale), _mm256_mulhi_epu16(far_hi, far_scale));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), _mm256_packus_epi16(lo, hi));
	}
	_mm256_zeroupper();
	return x;
}

// The weighted sum fits in an unsigned word (at most 255 * 255). In this
// range floor(sum / 255) = (sum * 0x8081u) >> 23, using mulhi then >> 7.
static UV_KERNEL("sse2") void OverlaySse2(BYTE* dst, const PixelRGBA* overlay, int width)
{
	const __m128i byte_mask = _mm_set1_epi16(255);
	const __m128i first_pixel = _mm_setr_epi32(0xffffff, 0, 0xffffff, 0);
	const __m128i second_pixel = _mm_setr_epi32(0, 0xffffff, 0, 0xffffff);
	for (; width >= 16; width -= 16, dst += 48, overlay += 16)
	{
		const auto* src = reinterpret_cast<const __m128i*>(overlay);
		const __m128i a = _mm_or_si128(_mm_loadu_si128(src), _mm_loadu_si128(src + 1));
		const __m128i b = _mm_or_si128(_mm_loadu_si128(src + 2), _mm_loadu_si128(src + 3));
		const __m128i clear = _mm_cmpeq_epi8(_mm_or_si128(a, b), _mm_setzero_si128());
		// Every fourth byte is alpha. Skip the block only when all 16 are zero.
		if ((_mm_movemask_epi8(clear) & 0x8888) == 0x8888)
			continue;

		for (int x = 0; x < 16; x += 4)
		{
			BYTE* out = dst + 3 * x;
			// Load exactly 12 BGR bytes, then expand two pixels in each 64-bit lane.
			const __m128i head = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(out));
			const __m128i tail = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(out + 4));
			const __m128i pairs = _mm_unpacklo_epi64(head, _mm_srli_epi64(tail, 16));
			const __m128i bg = _mm_or_si128(_mm_and_si128(pairs, first_pixel),
				_mm_and_si128(_mm_slli_epi64(pairs, 8), second_pixel));
			const __m128i fg = _mm_loadu_si128(reinterpret_cast<const __m128i*>(overlay + x));
			const __m128i a32 = _mm_srli_epi32(fg, 24);
			const __m128i alpha = _mm_or_si128(a32, _mm_slli_epi32(a32, 16));
			const __m128i inv = _mm_sub_epi16(byte_mask, alpha);
			auto blend = [&](__m128i back, __m128i front) UV_TARGET("sse2") {
				const __m128i sum = _mm_add_epi16(_mm_mullo_epi16(back, inv), _mm_mullo_epi16(front, alpha));
				return _mm_srli_epi16(_mm_mulhi_epu16(sum, _mm_set1_epi16(static_cast<short>(0x8081))), 7);
			};
			const __m128i rb = _mm_or_si128(_mm_srli_epi32(fg, 16), _mm_slli_epi32(fg, 16));
			const __m128i br_out = blend(_mm_and_si128(bg, byte_mask), _mm_and_si128(rb, byte_mask));
			const __m128i g_out = blend(_mm_srli_epi16(bg, 8), _mm_srli_epi16(fg, 8));
			const __m128i bgr = _mm_or_si128(br_out, _mm_slli_epi16(g_out, 8));
			const __m128i packed = _mm_or_si128(_mm_and_si128(bgr, first_pixel),
				_mm_srli_epi64(_mm_and_si128(bgr, second_pixel), 8));
			_mm_storel_epi64(reinterpret_cast<__m128i*>(out), packed);
			const __m128i last = _mm_or_si128(_mm_srli_epi64(packed, 32),
				_mm_slli_epi64(_mm_unpackhi_epi64(packed, packed), 16));
			_mm_storel_epi64(reinterpret_cast<__m128i*>(out + 4), last);
		}
	}
	OverlayScalar(dst, overlay, width);
}

static UV_KERNEL("avx2") void OverlayAvx2(BYTE* dst, const PixelRGBA* overlay, int width)
{
	const __m256i byte_mask = _mm256_set1_epi16(255);
	// Two overlapping loads cover exactly 24 BGR bytes. Each shuffle expands four pixels to RGBx.
	const __m128i first_four = _mm_setr_epi8(2, 1, 0, -1,  5, 4, 3, -1,  8, 7, 6, -1,  11, 10, 9, -1);
	const __m128i last_four = _mm_setr_epi8(6, 5, 4, -1,  9, 8, 7, -1,  12, 11, 10, -1,  15, 14, 13, -1);
	const __m256i alpha_words = _mm256_broadcastsi128_si256(
		_mm_setr_epi8(3, -1, 3, -1,  7, -1, 7, -1,  11, -1, 11, -1,  15, -1, 15, -1));
	const __m256i pack_bgr = _mm256_broadcastsi128_si256(
		_mm_setr_epi8(2, 1, 0,  6, 5, 4,  10, 9, 8,  14, 13, 12,  -1, -1, -1, -1));
	for (; width >= 16; width -= 16, dst += 48, overlay += 16)
	{
		const auto* src = reinterpret_cast<const __m256i*>(overlay);
		const __m256i combined = _mm256_or_si256(_mm256_loadu_si256(src), _mm256_loadu_si256(src + 1));
		if (_mm256_testz_si256(combined, _mm256_slli_epi32(byte_mask, 24)))
			continue;

		for (int x = 0; x < 16; x += 8)
		{
			BYTE* out = dst + 3 * x;
			const __m128i head = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(out)), first_four);
			const __m128i tail = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(out + 8)), last_four);
			const __m256i bg = _mm256_inserti128_si256(_mm256_castsi128_si256(head), tail, 1);
			const __m256i fg = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(overlay + x));
			const __m256i alpha = _mm256_shuffle_epi8(fg, alpha_words);
			const __m256i inv = _mm256_sub_epi16(byte_mask, alpha);
			auto blend = [&](__m256i back, __m256i front) UV_TARGET("avx2") {
				const __m256i sum = _mm256_add_epi16(_mm256_mullo_epi16(back, inv), _mm256_mullo_epi16(front, alpha));
				return _mm256_srli_epi16(_mm256_mulhi_epu16(sum, _mm256_set1_epi16(static_cast<short>(0x8081))), 7);
			};
			const __m256i rb = blend(_mm256_and_si256(bg, byte_mask), _mm256_and_si256(fg, byte_mask));
			const __m256i g = blend(_mm256_srli_epi16(bg, 8), _mm256_srli_epi16(fg, 8));
			const __m256i packed = _mm256_shuffle_epi8(_mm256_or_si256(rb, _mm256_slli_epi16(g, 8)), pack_bgr);
			const __m128i lo = _mm256_castsi256_si128(packed), hi = _mm256_extracti128_si256(packed, 1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm_or_si128(lo, _mm_slli_si128(hi, 12)));
			_mm_storel_epi64(reinterpret_cast<__m128i*>(out + 16), _mm_srli_si128(hi, 4));
		}
	}
	_mm256_zeroupper();
	OverlayScalar(dst, overlay, width);
}

#undef UV_TARGET
#undef UV_KERNEL
#endif

struct PixelKernels
{
	DecayInterior decay;
	OverlayRow overlay;
};

static PixelKernels SelectPixelKernels(int cpu_flags)
{
#ifdef ULTRAVIS_X86
	if (cpu_flags & CPUF_AVX2)
		return { DecayAvx2, OverlayAvx2 };
	if (cpu_flags & CPUF_SSE2)
		return { DecaySse2, OverlaySse2 };
#endif
	return { nullptr, OverlayScalar };
}

// Helper function for color calculation of the frequency splines via cosine functions
void
GetSpectrumColor(float progress, BYTE& r, BYTE& g, BYTE& b)
{
	using std::cos;
	// progress runs from 0.0 (bottom/low frequencies) to 1.0 (top/high frequencies)
	float angle = (1.0f - progress) * 5.0f + ((float)M_PI * 2.0f - 5.0f);
	r = static_cast<BYTE>(clamp((cos(angle) * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
	g = static_cast<BYTE>(clamp((cos(angle + 2.0f * (float)M_PI / 3.0f) * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
	b = static_cast<BYTE>(clamp((cos(angle + 4.0f * (float)M_PI / 3.0f) * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
}

// Catmull-Rom Spline Interpolation (reuses output vectors without reallocating capacity)
void
InterpolateSpline(const std::vector<float>& x_pts, const std::vector<float>& y_pts, int num_output, std::vector<float>& out_x, std::vector<float>& out_y)
{
	out_x.clear();
	out_y.clear();

	int n = (int)x_pts.size();
	if (n < 2 || num_output < 2)
		return;

	out_x.reserve(num_output);
	out_y.reserve(num_output);

	for (int i = 0; i < num_output; ++i)
	{
		float t_global = (float)i / (num_output - 1) * (n - 1);
		int idx = (int)std::floor(t_global);
		if (idx >= n - 1)
			idx = n - 2;
		float t = t_global - idx;

		// Determine control points
		int p0 = (std::max)(0, idx - 1);
		int p1 = idx;
		int p2 = idx + 1;
		int p3 = (std::min)(n - 1, idx + 2);

		auto catmull_rom = [](float v0, float v1, float v2, float v3, float t) { return 0.5f * ((2.0f * v1) + (-v0 + v2) * t + (2.0f * v0 - 5.0f * v1 + 4.0f * v2 - v3) * t * t + (-v0 + 3.0f * v1 - 3.0f * v2 + v3) * t * t * t); };

		out_x.push_back(catmull_rom(x_pts[p0], x_pts[p1], x_pts[p2], x_pts[p3], t));
		out_y.push_back(catmull_rom(y_pts[p0], y_pts[p1], y_pts[p2], y_pts[p3], t));
	}
}

// Hann-windowed magnitude spectrum of two real channels, exact N-point DFT for any N (Bluestein, O(N log N))
class SpectrumPlan
{
	typedef std::complex<double> cplx;
	int N, L;
	std::vector<float> hann;
	std::vector<cplx> chirp;      // exp(-i*pi*n^2/N)
	std::vector<cplx> kernel_fft; // FFT of the conjugated chirp, length L
	std::vector<cplx> roots;      // exp(-2*pi*i*j/L), j < L/2
	std::vector<cplx> work;

	// In-place radix-2 FFT of length L; the inverse is unscaled
	void
		Fft(std::vector<cplx>& a, bool inverse) const
	{
		for (int i = 1, j = 0; i < L; ++i)
		{
			int bit = L >> 1;
			for (; j & bit; bit >>= 1)
				j ^= bit;
			j ^= bit;
			if (i < j)
				std::swap(a[i], a[j]);
		}
		for (int len = 2; len <= L; len <<= 1)
		{
			int half = len >> 1;
			int stride = L / len;
			for (int i = 0; i < L; i += len)
			{
				for (int k = 0; k < half; ++k)
				{
					cplx w = inverse ? std::conj(roots[k * stride]) : roots[k * stride];
					cplx u = a[i + k];
					cplx v = a[i + k + half] * w;
					a[i + k] = u + v;
					a[i + k + half] = u - v;
				}
			}
		}
	}

public:
	explicit SpectrumPlan(int num_samples) : N(num_samples), L(1)
	{
		while (L < 2 * N - 1)
			L <<= 1;

		// Same window expression as the reference DFT
		hann.resize(N);
		for (int i = 0; i < N; ++i)
			hann[i] = 0.5f * (1.0f - cos(2.0f * (float)M_PI * i / (N - 1)));

		chirp.resize(N);
		for (int n = 0; n < N; ++n)
		{
			long long sq = ((long long)n * n) % (2LL * N);
			chirp[n] = std::polar(1.0, -M_PI * (double)sq / N);
		}

		roots.resize(L / 2);
		for (int j = 0; j < L / 2; ++j)
			roots[j] = std::polar(1.0, -2.0 * M_PI * j / L);

		kernel_fft.assign(L, cplx(0.0, 0.0));
		kernel_fft[0] = std::conj(chirp[0]);
		for (int n = 1; n < N; ++n)
			kernel_fft[n] = kernel_fft[L - n] = std::conj(chirp[n]);
		Fft(kernel_fft, false);
		work.resize(L);
	}

	// Magnitudes |X[k]| / N for k < N/2 of both channels, packed as one complex signal left + i*right
	void
		Compute(const float* left, const float* right, std::vector<float>& mag_l, std::vector<float>& mag_r)
	{
		int M = N / 2;
		for (int n = 0; n < N; ++n)
			work[n] = cplx(left[n] * hann[n], right[n] * hann[n]) * chirp[n];
		std::fill(work.begin() + N, work.end(), cplx(0.0, 0.0));
		Fft(work, false);
		for (int i = 0; i < L; ++i)
			work[i] *= kernel_fft[i];
		Fft(work, true);
		double scale = 1.0 / L;
		for (int k = 0; k < N; ++k)
			work[k] *= chirp[k] * scale;

		mag_l.resize(M);
		mag_r.resize(M);
		for (int k = 0; k < M; ++k)
		{
			cplx zk = work[k];
			cplx zc = std::conj(work[(N - k) % N]);
			mag_l[k] = (float)(std::abs(zk + zc) * 0.5 / N);
			mag_r[k] = (float)(std::abs(zk - zc) * 0.5 / N);
		}
	}
};

// Persistent worker threads that split a row range into contiguous parts
class RowPool
{
	int count;
	std::vector<std::thread> workers;
	std::mutex mutex;
	std::condition_variable start, done;
	const std::function<void(int, int)>* job = nullptr;
	int job_rows = 0, job_parts = 0, pending = 0;
	unsigned generation = 0;
	bool stop = false;

	void
		Worker(int id)
	{
		unsigned seen = 0;
		for (;;)
		{
			const std::function<void(int, int)>* fn;
			int rows, parts;
			{
				std::unique_lock<std::mutex> lock(mutex);
				start.wait(lock, [&] { return stop || generation != seen; });
				if (stop)
					return;
				seen = generation;
				fn = job;
				rows = job_rows;
				parts = job_parts;
			}
			if (id >= parts)
				continue;
			(*fn)(rows * id / parts, rows * (id + 1) / parts);
			std::lock_guard<std::mutex> lock(mutex);
			if (--pending == 0)
				done.notify_one();
		}
	}

public:
	explicit RowPool(int threads) : count((std::max)(1, threads))
	{
		for (int i = 1; i < count; ++i)
			workers.emplace_back([this, i] { Worker(i); });
	}

	~RowPool()
	{
		{
			std::lock_guard<std::mutex> lock(mutex);
			stop = true;
		}
		start.notify_all();
		for (auto& t : workers)
			t.join();
	}

	void
		Run(int rows, const std::function<void(int, int)>& fn)
	{
		int parts = (std::min)(count, rows);
		if (parts <= 1)
		{
			fn(0, rows);
			return;
		}
		{
			std::lock_guard<std::mutex> lock(mutex);
			job = &fn;
			job_rows = rows;
			job_parts = parts;
			pending = parts - 1;
			++generation;
		}
		start.notify_all();
		fn(0, rows / parts);
		std::unique_lock<std::mutex> lock(mutex);
		done.wait(lock, [&] { return pending == 0; });
	}
};

class UltraVisualizerMT : public GenericVideoFilter
{
	// Frames replayed after a seek; the alpha channel of the trails reaches 0 after about 14 frames
	static const int WARMUP_FRAMES = 16;
	// Largest forward gap that is rendered and kept instead of resyncing (out-of-order requests under Prefetch)
	static const int MAX_AHEAD = 32;
	// Maximum cached FFT plans to prevent unbounded memory growth on VFR clips
	static const size_t MAX_PLANS = 16;

	int width, height;
	float line_thickness;
	const PixelKernels kernels;
	std::vector<PixelRGBA> internal_buffer; // Internal storage clip in RGBA32 format
	std::vector<PixelRGBA> next_buffer;
	std::map<int, SpectrumPlan> plans;
	std::map<int, PVideoFrame> ahead; // Frames rendered ahead of a request
	int next_n = 0;                   // Frame whose audio the internal buffer expects next
	RowPool pool;

	// Reusable scratch buffers to prevent per-frame heap allocations
	std::vector<float> audio_buf;
	std::vector<float> left_chan;
	std::vector<float> right_chan;
	std::vector<float> fft_left;
	std::vector<float> fft_right;
	std::vector<float> x_left_pts, y_left_pts, x_right_pts, y_right_pts;
	std::vector<float> s_x_l, s_y_l, s_x_r, s_y_r;
	std::vector<float> phase_x_pts, phase_y_pts;
	std::vector<float> s_p_x, s_p_y;

#ifdef ULTRAVIS_REFERENCE_DFT
	// Discrete Fourier Transform (DFT) as a standalone FFT replacement for an audio sample window
	void
		ComputeDFT(const float* samples, int num_samples, std::vector<float>& magnitude)
	{
		int N = num_samples;
		int M = N / 2;
		magnitude.assign(M, 0.0f);
		if (N <= 0)
			return;

		std::vector<float> windowed(N);
		for (int i = 0; i < N; ++i)
		{
			float hann = 0.5f * (1.0f - cos(2.0f * (float)M_PI * i / (N - 1)));
			windowed[i] = samples[i] * hann;
		}

		// Twiddle table: cos/sin of 2*pi*j/N for j = 0..N-1
		std::vector<float> cos_tab(N), sin_tab(N);
		for (int j = 0; j < N; ++j)
		{
			double angle = 2.0 * M_PI * j / N;
			cos_tab[j] = (float)cos(angle);
			sin_tab[j] = (float)sin(angle);
		}

		for (int k = 0; k < M; ++k)
		{
			float real = 0.0f;
			float imag = 0.0f;
			int idx = 0; // (k * n) mod N
			for (int n = 0; n < N; ++n)
			{
				real += windowed[n] * cos_tab[idx];
				imag -= windowed[n] * sin_tab[idx];
				idx += k;
				if (idx >= N)
					idx -= N;
			}
			magnitude[k] = std::sqrt(real * real + imag * imag) / N;
		}
	}
#endif

	void
		ComputeSpectra(const std::vector<float>& left, const std::vector<float>& right, int num_samples, std::vector<float>& out_fft_left, std::vector<float>& out_fft_right)
	{
#ifdef ULTRAVIS_REFERENCE_DFT
		ComputeDFT(left.data(), num_samples, out_fft_left);
		ComputeDFT(right.data(), num_samples, out_fft_right);
#else
		auto it = plans.find(num_samples);
		if (it == plans.end())
		{
			if (plans.size() >= MAX_PLANS)
				plans.clear();
			it = plans.emplace(num_samples, SpectrumPlan(num_samples)).first;
		}
		it->second.Compute(left.data(), right.data(), out_fft_left, out_fft_right);
#endif
	}

	// Function to draw a line with variable thickness using Xiaolin Wu's anti-aliasing algorithm, linear color gradient and maximum blending
	void
		DrawLineAA(float x0, float y0, float x1, float y1, BYTE r0, BYTE g0, BYTE b0, BYTE r1, BYTE g1, BYTE b1, BYTE a)
	{
		if (line_thickness <= 0.0f)
			return;

		auto plot = [&](int x, int y, float alpha_factor, float t) {
			if (x >= 0 && x < width && y >= 0 && y < height && alpha_factor > 0.0f)
			{
				PixelRGBA& dest = internal_buffer[static_cast<size_t>(y) * width + x];
				BYTE target_a = static_cast<BYTE>(clamp(a * alpha_factor, 0.0f, 255.0f));
				BYTE r = static_cast<BYTE>(clamp(r0 + (r1 - r0) * t + 0.5f, 0.0f, 255.0f));
				BYTE g = static_cast<BYTE>(clamp(g0 + (g1 - g0) * t + 0.5f, 0.0f, 255.0f));
				BYTE b = static_cast<BYTE>(clamp(b0 + (b1 - b0) * t + 0.5f, 0.0f, 255.0f));

				// Blending using maximum for each channel separately
				dest.r = (std::max)(dest.r, r);
				dest.g = (std::max)(dest.g, g);
				dest.b = (std::max)(dest.b, b);
				dest.a = (std::max)(dest.a, target_a);
			}
		};

		auto ipart = [](float x) -> int { return static_cast<int>(std::floor(x)); };
		auto fpart = [](float x) -> float { return x - std::floor(x); };
		auto rfpart = [&](float x) -> float { return 1.0f - fpart(x); };

		bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
		if (steep)
		{
			std::swap(x0, y0);
			std::swap(x1, y1);
		}
		if (x0 > x1)
		{
			std::swap(x0, x1);
			std::swap(y0, y1);
			std::swap(r0, r1);
			std::swap(g0, g1);
			std::swap(b0, b1);
		}

		float dx = x1 - x0;
		float dy = y1 - y0;
		float gradient = (dx == 0.0f) ? 1.0f : dy / dx;

		// Adjust thickness perpendicular to the dominant axis to preserve perceived stroke width
		float cos_theta = 1.0f / std::sqrt(1.0f + gradient * gradient);
		float half_w = 0.5f * (line_thickness / cos_theta);

		auto draw_slice = [&](int x, float y_center, float x_weight, float t) {
			if (x_weight <= 0.0f)
				return;
			float y_top = y_center - half_w;
			float y_bot = y_center + half_w;
			int y_start = ipart(y_top);
			int y_end = ipart(y_bot);

			if (y_start == y_end)
			{
				float cov = (y_bot - y_top) * x_weight;
				if (steep)
					plot(y_start, x, cov, t);
				else
					plot(x, y_start, cov, t);
				return;
			}

			float cov_top = rfpart(y_top) * x_weight;
			float cov_bot = fpart(y_bot) * x_weight;

			if (steep)
			{
				plot(y_start, x, cov_top, t);
				for (int y = y_start + 1; y < y_end; ++y)
					plot(y, x, x_weight, t);
				plot(y_end, x, cov_bot, t);
			}
			else
			{
				plot(x, y_start, cov_top, t);
				for (int y = y_start + 1; y < y_end; ++y)
					plot(x, y, x_weight, t);
				plot(x, y_end, cov_bot, t);
			}
		};

		// Handle first endpoint
		float xend = std::round(x0);
		float yend = y0 + gradient * (xend - x0);
		float xgap = rfpart(x0 + 0.5f);
		int xpxl1 = static_cast<int>(xend);
		draw_slice(xpxl1, yend, xgap, 0.0f);

		float intery = yend + gradient; // First y-intersection for the main loop

		// Handle second endpoint
		xend = std::round(x1);
		yend = y1 + gradient * (xend - x1);
		xgap = fpart(x1 + 0.5f);
		int xpxl2 = static_cast<int>(xend);
		draw_slice(xpxl2, yend, xgap, 1.0f);

		// Main loop with linear color interpolation
		float span = static_cast<float>(xpxl2 - xpxl1);
		float inv_span = (span > 0.0f) ? (1.0f / span) : 0.0f;

		for (int x = xpxl1 + 1; x <= xpxl2; ++x)
		{
			float t = clamp((x - xpxl1) * inv_span, 0.0f, 1.0f);
			draw_slice(x, intery, 1.0f, t);
			intery += gradient;
		}
	}

	// --- 1. + 2. Draw the audio of frame n into the internal buffer ---
	void
		DrawAudio(int n, IScriptEnvironment* env)
	{
		if (!vi.HasAudio())
			return;

		// Determine the audio time window for this video frame
		__int64 start_sample = (__int64)n * vi.audio_samples_per_second * vi.fps_denominator / vi.fps_numerator;
		__int64 end_sample = (__int64)(n + 1) * vi.audio_samples_per_second * vi.fps_denominator / vi.fps_numerator;
		int num_samples = static_cast<int>(end_sample - start_sample);

		if (num_samples <= 32) // Minimum number for a meaningful FFT
			return;

		audio_buf.resize(static_cast<size_t>(num_samples) * vi.nchannels);
		child->GetAudio(audio_buf.data(), start_sample, num_samples, env);

		left_chan.resize(num_samples);
		right_chan.resize(num_samples);
		for (int i = 0; i < num_samples; ++i)
		{
			left_chan[i] = audio_buf[i * vi.nchannels + 0];
			right_chan[i] = audio_buf[i * vi.nchannels + 1];
		}

		ComputeSpectra(left_chan, right_chan, num_samples, fft_left, fft_right);

		// --- 1. Left and right border: Draw frequency splines ---
		int bands = (int)fft_left.size();
		if (bands < 2)
			return;

		x_left_pts.clear();
		y_left_pts.clear();
		x_right_pts.clear();
		y_right_pts.clear();

		x_left_pts.reserve(bands);
		y_left_pts.reserve(bands);
		x_right_pts.reserve(bands);
		y_right_pts.reserve(bands);

		for (int i = 0; i < bands; ++i)
		{
			float progress = (float)i / (bands - 1);
			// Square stretching for vertical distribution (low frequencies bottom, high top)
			float norm_y = std::sqrt(progress);
			float y_pos = norm_y * (height - 1);

			// Maximum displacement is 1/2 of the total width
			float max_amplitude = (width / 4.0f);
			float amp_l = clamp(std::sqrt(std::sqrt(fft_left[i])) * max_amplitude, 0.0f, max_amplitude);
			float amp_r = clamp(std::sqrt(std::sqrt(fft_right[i])) * max_amplitude, 0.0f, max_amplitude);

			x_left_pts.push_back(amp_l);
			y_left_pts.push_back(y_pos);

			x_right_pts.push_back((width - 1) - amp_r);
			y_right_pts.push_back(y_pos);
		}

		// Generation of interpolated points for smooth splines
		InterpolateSpline(x_left_pts, y_left_pts, height * 2, s_x_l, s_y_l);
		InterpolateSpline(x_right_pts, y_right_pts, height * 2, s_x_r, s_y_r);

		// Draw splines onto the internal buffer (with size_t underflow protection and linear gradient)
		if (s_x_l.size() > 1 && s_x_r.size() > 1)
		{
			size_t count = (std::min)(s_x_l.size(), s_x_r.size());
			for (size_t i = 0; i < count - 1; ++i)
			{
				float prog0 = (float)i / count;
				float prog1 = (float)(i + 1) / count;
				BYTE r0, g0, b0, r1, g1, b1;
				GetSpectrumColor(prog0, r0, g0, b0);
				GetSpectrumColor(prog1, r1, g1, b1);
				DrawLineAA(s_x_l[i], s_y_l[i], s_x_l[i + 1], s_y_l[i + 1], r0, g0, b0, r1, g1, b1, 255);
				DrawLineAA(s_x_r[i], s_y_r[i], s_x_r[i + 1], s_y_r[i + 1], r0, g0, b0, r1, g1, b1, 255);
			}
		}

		// --- 2. Middle: Phase display ---
		phase_x_pts.clear();
		phase_y_pts.clear();

		int step = (std::max)(1, num_samples / 256); // Limiting control points for a smooth spline
		size_t est_pts = static_cast<size_t>(num_samples / step) + 1;
		phase_x_pts.reserve(est_pts);
		phase_y_pts.reserve(est_pts);

		float cx = width / 2.0f;
		float cy = height / 2.0f;
		float max_r = (std::min)(cx, cy) * 0.9f; // Scaling radius for the center

		for (int i = 0; i < num_samples; i += step)
		{
			float L = left_chan[i];
			float R = right_chan[i];

			// Horizontal position: Sine of the difference (R - L)
			float px = cx + sin(R - L) * cx * 0.6f;
			// Vertical position: Sine of the sum (L + R)
			float py = cy + sin(L + R) * cy * 0.8f;
			phase_x_pts.push_back(px);
			phase_y_pts.push_back(py);
		}

		InterpolateSpline(phase_x_pts, phase_y_pts, num_samples, s_p_x, s_p_y);

		if (s_p_x.size() > 1)
		{
			auto get_phase_color = [&](float px, float py, BYTE& r, BYTE& g, BYTE& b) {
				float dist_x = std::abs(px - cx) / cx;
				float dist_y = std::abs(py - cy) / cy;
				float dist_center = std::sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy)) / max_r;
				r = static_cast<BYTE>(clamp(dist_y * 255.0f, 0.0f, 255.0f));
				g = static_cast<BYTE>(clamp(std::sqrt((std::max)(0.0f, 1.0f - dist_center)) * 255.0f, 0.0f, 255.0f));
				b = static_cast<BYTE>(clamp(dist_x * 255.0f, 0.0f, 255.0f));
			};

			BYTE r0, g0, b0, r1, g1, b1;
			get_phase_color(s_p_x[0], s_p_y[0], r0, g0, b0);

			for (size_t i = 0; i < s_p_x.size() - 1; ++i)
			{
				get_phase_color(s_p_x[i + 1], s_p_y[i + 1], r1, g1, b1);
				DrawLineAA(s_p_x[i], s_p_y[i], s_p_x[i + 1], s_p_y[i + 1], r0, g0, b0, r1, g1, b1, 255);
				r0 = r1;
				g0 = g1;
				b0 = b1;
			}
		}
	}

	// --- 3. Overlay onto target video (Input video in RGB24 format) ---
	void
		ApplyOverlay(PVideoFrame& src)
	{
		BYTE* video_ptr = src->GetWritePtr();
		int pitch = src->GetPitch();
		pool.Run(height, [&](int y0, int y1) {
			for (int y = y0; y < y1; ++y)
			{
				BYTE* row = video_ptr + static_cast<size_t>(y) * pitch;
				const PixelRGBA* over_row = &internal_buffer[static_cast<size_t>(y) * width];
				kernels.overlay(row, over_row, width);
			}
		});
	}

	// --- 4. Pixel feedback/decay of the internal clip (Liquid blur attenuation) ---
	void
		DecayBuffer()
	{
		auto blend = [](const PixelRGBA& top_left, const PixelRGBA& top_right, const PixelRGBA& bottom_left, const PixelRGBA& bottom_right,
			const PixelRGBA& top2, const PixelRGBA& bottom2, const PixelRGBA& left2, const PixelRGBA& right2, PixelRGBA& out) {
				out.r = (BYTE)(std::min)((top_left.r + top_right.r + bottom_left.r + bottom_right.r) / 8 + (top2.r + bottom2.r + left2.r + right2.r) / 8, 255);
				out.g = (BYTE)(std::min)((top_left.g + top_right.g + bottom_left.g + bottom_right.g) / 8 + (top2.g + bottom2.g + left2.g + right2.g) / 8, 255);
				out.b = (BYTE)(std::min)((top_left.b + top_right.b + bottom_left.b + bottom_right.b) / 8 + (top2.b + bottom2.b + left2.b + right2.b) / 8, 255);
				out.a = (BYTE)(std::min)((top_left.a + top_right.a + bottom_left.a + bottom_right.a) / 10 + (top2.a + bottom2.a + left2.a + right2.a) / 12, 255);
			};

		pool.Run(height, [&](int y0, int y1) {
			for (int y = y0; y < y1; ++y)
			{
				const PixelRGBA* row = &internal_buffer[static_cast<size_t>(y) * width];
				const PixelRGBA* up1 = &internal_buffer[static_cast<size_t>(clamp(y - 1, 0, height - 1)) * width];
				const PixelRGBA* up2 = &internal_buffer[static_cast<size_t>(clamp(y - 2, 0, height - 1)) * width];
				const PixelRGBA* dn1 = &internal_buffer[static_cast<size_t>(clamp(y + 1, 0, height - 1)) * width];
				const PixelRGBA* dn2 = &internal_buffer[static_cast<size_t>(clamp(y + 2, 0, height - 1)) * width];
				PixelRGBA* out = &next_buffer[static_cast<size_t>(y) * width];

				auto edge = [&](int x) {
					int l1 = clamp(x - 1, 0, width - 1), l2 = clamp(x - 2, 0, width - 1);
					int r1 = clamp(x + 1, 0, width - 1), r2 = clamp(x + 2, 0, width - 1);
					blend(up1[l1], up1[r1], dn1[l1], dn1[r1], up2[x], dn2[x], row[l2], row[r2], out[x]);
				};

				int inner_end = width - 2;
				for (int x = 0; x < (std::min)(2, width); ++x)
					edge(x);
				int x = kernels.decay ? kernels.decay(row, up1, dn1, up2, dn2, out, inner_end) : 2;
				for (; x < inner_end; ++x)
					blend(up1[x - 1], up1[x + 1], dn1[x - 1], dn1[x + 1], up2[x], dn2[x], row[x - 2], row[x + 2], out[x]);
				for (int x = (std::max)(2, inner_end); x < width; ++x)
					edge(x);
			}
		});
		internal_buffer.swap(next_buffer);
	}

	// Render frame n and advance the internal buffer to frame n + 1
	PVideoFrame
		RenderFrame(int n, IScriptEnvironment* env)
	{
		PVideoFrame src = child->GetFrame(n, env);
		env->MakeWritable(&src);
		DrawAudio(n, env);
		ApplyOverlay(src);
		DecayBuffer();
		next_n = n + 1;
		return src;
	}

	// Rebuild the internal buffer for frame n from the preceding frames' audio
	void
		Resync(int n, IScriptEnvironment* env)
	{
		ahead.clear();
		std::fill(internal_buffer.begin(), internal_buffer.end(), PixelRGBA{ 0, 0, 0, 0 });
		if (vi.HasAudio())
		{
			for (int m = (std::max)(0, n - WARMUP_FRAMES); m < n; ++m)
			{
				DrawAudio(m, env);
				DecayBuffer();
			}
		}
		next_n = n;
	}

	static int
		AutoThreads(int threads)
	{
		if (threads > 0)
			return threads;
		int hw = (int)std::thread::hardware_concurrency();
		return clamp(hw, 1, 8);
	}

public:
	UltraVisualizerMT(PClip _child, int threads, float thickness, IScriptEnvironment* env)
		: GenericVideoFilter(_child), line_thickness(clamp(thickness, 0.1f, 64.0f)),
		  kernels(SelectPixelKernels(env->GetCPUFlags())), pool(AutoThreads(threads))
	{
		if (!vi.HasVideo())
		{
			env->ThrowError("Ultra Visualizer: The clip must contain video!");
		}

		if (thickness <= 0.0f)
		{
			env->ThrowError("Ultra Visualizer: thickness must be greater than 0.0!");
		}

		// If the video is not in the required RGB24 format, force/convert it
		if (!vi.IsRGB24())
		{
			AVSValue cargs[1] = { child };
			PClip converted = env->Invoke("ConvertToRGB24", AVSValue(cargs, 1)).AsClip();
			child = converted;
			vi.pixel_type = VideoInfo::CS_BGR24;
		}

		// Verify or convert audio formats (Requires Float)
		if (vi.HasAudio() && vi.SampleType() != SAMPLE_FLOAT)
		{
			AVSValue cargs[1] = { child };
			PClip convertedAudio = env->Invoke("ConvertAudioToFloat", AVSValue(cargs, 1)).AsClip();
			child = convertedAudio;
			vi.sample_type = SAMPLE_FLOAT;
		}

		// If audio is mono, duplicate to stereo via MergeChannels
		if (vi.HasAudio() && vi.nchannels == 1)
		{
			AVSValue args[2] = { child, child };
			PClip stereoAudio = env->Invoke("MergeChannels", AVSValue(args, 2)).AsClip();
			child = stereoAudio;
			vi.nchannels = 2;
		}

		width = vi.width;
		height = vi.height;

		// Safe size_t multiplication to avoid signed integer overflow on huge resolutions
		size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
		internal_buffer.resize(pixel_count, PixelRGBA{ 0, 0, 0, 0 });
		next_buffer.resize(pixel_count);
	}

	// The internal buffer carries state from frame to frame, so calls must not run concurrently
	int __stdcall SetCacheHints(int cachehints, int frame_range) override
	{
		return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
	{
		auto it = ahead.find(n);
		if (it != ahead.end())
		{
			PVideoFrame frame = it->second;
			ahead.erase(it);
			return frame;
		}

		if (n < next_n || n - next_n > MAX_AHEAD)
			Resync(n, env);

		// Frames skipped by an out-of-order request are rendered now and kept for their own request
		while (next_n < n)
		{
			int m = next_n;
			ahead[m] = RenderFrame(m, env);
		}
		while (!ahead.empty() && ahead.begin()->first < n - MAX_AHEAD)
			ahead.erase(ahead.begin());

		return RenderFrame(n, env);
	}
};

// Factory function for AviSynth+
AVSValue Create_UltraVisMT(AVSValue args, void* user_data, IScriptEnvironment* env)
{
	return new UltraVisualizerMT(
		args[0].AsClip(),
		args[1].AsInt(0),
		static_cast<float>(args[2].AsFloat(2.0)),
		env
	);
}

// Plugin registration when loading the DLL
const AVS_Linkage* AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("UltraVisMT", "c[threads]i[thickness]f", Create_UltraVisMT, 0);
	return "Ultra Visualizer multi-threaded";
}
