#include <windows.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include "avisynth.h"

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

// Helper function for color calculation of the frequency splines via cosine functions
void
GetSpectrumColor(float progress, BYTE& r, BYTE& g, BYTE& b)
{
	// progress runs from 0.0 (bottom/low frequencies) to 1.0 (top/high frequencies)
	float angle = (1.0f - progress) * 5.0f + ((float)M_PI * 2.0f - 5.0f);
	r = static_cast<BYTE>(clamp((cos(angle) * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
	g = static_cast<BYTE>(clamp((cos(angle + 2.0f * (float)M_PI / 3.0f) * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
	b = static_cast<BYTE>(clamp((cos(angle + 4.0f * (float)M_PI / 3.0f) * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
}

// Catmull-Rom Spline Interpolation
void
InterpolateSpline(const std::vector<float>& x_pts, const std::vector<float>& y_pts, int num_output, std::vector<float>& out_x, std::vector<float>& out_y)
{
	int n = (int)x_pts.size();
	if (n < 2)
		return;

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

class UltraVisualizer : public GenericVideoFilter
{
	int width, height;
	std::vector<PixelRGBA> internal_buffer; // Internal storage clip in RGBA32 format

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

	// Function to draw a line with 2px thickness, anti-aliasing, and maximum blending
	void
		DrawLineAA(int x0, int y0, int x1, int y1, BYTE r, BYTE g, BYTE b, BYTE a)
	{
		int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
		int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
		int err = dx - dy, e2;
		int ed = dx + dy == 0 ? 1 : (int)sqrt((float)dx * dx + (float)dy * dy);

		auto plot = [&](int x, int y, float alpha_factor) {
			if (x >= 0 && x < width && y >= 0 && y < height)
			{
				PixelRGBA& dest = internal_buffer[y * width + x];
				BYTE target_a = static_cast<BYTE>(a * alpha_factor);

				// Blending using maximum for each channel separately
				dest.r = (std::max)(dest.r, r);
				dest.g = (std::max)(dest.g, g);
				dest.b = (std::max)(dest.b, b);
				dest.a = (std::max)(dest.a, target_a);
			}
			};

		// Extension for 2 pixels thickness through parallel line plots
		while (true)
		{
			float alpha = std::sqrt(1.0f - (ed == 0 ? 0.0f : (float)abs(err - dx + dy) / ed));
			alpha = clamp(alpha, 0.0f, 1.0f);

			plot(x0, y0, alpha);
			// Line thickening
			plot(x0, y0 - 1, alpha * 0.7f);
			plot(x0, y0 + 1, alpha * 0.7f);
			plot(x0 - 1, y0, alpha * 0.7f);
			plot(x0 + 1, y0, alpha * 0.7f);
			plot(x0 - 1, y0 - 1, alpha * 0.4f);
			plot(x0 - 1, y0 + 1, alpha * 0.4f);
			plot(x0 + 1, y0 - 1, alpha * 0.4f);
			plot(x0 + 1, y0 + 1, alpha * 0.4f);

			if (x0 == x1 && y0 == y1)
				break;
			e2 = 2 * err;
			if (e2 >= -dy)
			{
				err -= dy;
				x0 += sx;
			}
			if (e2 <= dx)
			{
				err += dx;
				y0 += sy;
			}
		}
	}

public:
	UltraVisualizer(PClip _child, IScriptEnvironment* env) : GenericVideoFilter(_child)
	{
		if (!vi.HasVideo())
		{
			env->ThrowError("Ultra Visualizer: The clip must contain video!");
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

		// Initialize the internal storage clip with transparency (Alpha = 0)
		internal_buffer.resize(width * height, PixelRGBA{ 0, 0, 0, 0 });
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
	{
		// Fetch the current video frame
		PVideoFrame src = child->GetFrame(n, env);
		env->MakeWritable(&src);

		// Determine the audio time window for this video frame
		if (vi.HasAudio())
		{
			__int64 start_sample = (__int64)n * vi.audio_samples_per_second * vi.fps_denominator / vi.fps_numerator;
			__int64 end_sample = (__int64)(n + 1) * vi.audio_samples_per_second * vi.fps_denominator / vi.fps_numerator;
			int num_samples = static_cast<int> (end_sample - start_sample);

			if (num_samples > 32)
			{ // Minimum number for a meaningful FFT
				std::vector<float> audio_buf(num_samples * vi.nchannels);
				child->GetAudio(audio_buf.data(), start_sample, num_samples, env);

				std::vector<float> left_chan(num_samples);
				std::vector<float> right_chan(num_samples);
				for (int i = 0; i < num_samples; ++i)
				{
					left_chan[i] = audio_buf[i * vi.nchannels + 0];
					right_chan[i] = audio_buf[i * vi.nchannels + 1];
				}

				std::vector<float> fft_left, fft_right;
				ComputeDFT(left_chan.data(), num_samples, fft_left);
				ComputeDFT(right_chan.data(), num_samples, fft_right);

				// --- 1. Left and right border: Draw frequency splines ---
				int bands = (int)fft_left.size();
				std::vector<float> x_left_pts, y_left_pts, x_right_pts, y_right_pts;

				for (int i = 0; i < bands; ++i)
				{
					float progress = (float)i / (bands - 1);
					// Square stretching for vertical distribution (low frequencies bottom, high top)
					// float norm_y = 1.0f - (progress * progress);
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
				std::vector<float> s_x_l, s_y_l, s_x_r, s_y_r;
				InterpolateSpline(x_left_pts, y_left_pts, height * 2, s_x_l, s_y_l);
				InterpolateSpline(x_right_pts, y_right_pts, height * 2, s_x_r, s_y_r);

				// Draw splines onto the internal buffer
				for (size_t i = 0; i < s_x_l.size() - 1; ++i)
				{
					float prog = (float)i / s_x_l.size();
					BYTE r, g, b;
					GetSpectrumColor(prog, r, g, b);
					DrawLineAA((int)s_x_l[i], (int)s_y_l[i], (int)s_x_l[i + 1], (int)s_y_l[i + 1], r, g, b, 255);
					DrawLineAA((int)s_x_r[i], (int)s_y_r[i], (int)s_x_r[i + 1], (int)s_y_r[i + 1], r, g, b, 255);
				}

				// --- 2. Middle: Phase display ---
				std::vector<float> phase_x_pts, phase_y_pts;
				int step = (std::max)(1, num_samples / 256); // Limiting control points for a smooth spline
				float cx = width / 2.0f;
				float cy = height / 2.0f;
				float max_r = (std::min)(cx, cy) * 0.9f; // Scaling radius for the center

				for (int i = 0; i < num_samples; i += step)
				{
					float L = left_chan[i];
					float R = right_chan[i];

					// Horizontal position: Sine of the difference (R - L)
					float px = cx + sin(R - L) * /*max_r*/ cx * 0.6f;
					// Vertical position: Sine of the sum (L + R)
					float py = cy + sin(L + R) * /*max_r*/ cy * 0.8f;
					phase_x_pts.push_back(px);
					phase_y_pts.push_back(py);
				}
				std::vector<float> s_p_x, s_p_y;
				InterpolateSpline(phase_x_pts, phase_y_pts, num_samples, s_p_x, s_p_y);
				for (size_t i = 0; i < s_p_x.size() - 1; ++i)
				{
					float px = s_p_x[i];
					float py = s_p_y[i];
					// Dynamic color calculation based on distance to the center
					float dist_x = std::abs(px - cx) / cx;
					float dist_y = std::abs(py - cy) / cy;
					float dist_center = std::sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy)) / (max_r);
					BYTE r = static_cast<BYTE>(clamp(dist_y * 255.0f, 0.0f, 255.0f));
					BYTE g = static_cast<BYTE>(clamp(std::sqrt(1.0f - dist_center) * 255.0f, 0.0f, 255.0f));
					BYTE b = static_cast<BYTE>(clamp(dist_x * 255.0f, 0.0f, 255.0f));
					DrawLineAA((int)px, (int)py, (int)s_p_x[i + 1], (int)s_p_y[i + 1], r, g, b, 255);
				}
			}
		}
		// --- 3. Overlay onto target video (Input video in RGB24 format) ---
		BYTE* video_ptr = src->GetWritePtr();
		int pitch = src->GetPitch();
		for (int y = 0; y < height; ++y)
		{
			BYTE* row = video_ptr + y * pitch;
			for (int x = 0; x < width; ++x)
			{
				const PixelRGBA& overlay = internal_buffer[y * width + x];
				if (overlay.a > 0)
				{
					float alpha_pct = overlay.a / 255.0f;
					// Avisynth+ uses BGR byte order in memory for RGB24
					row[x * 3 + 0] = static_cast<BYTE>(row[x * 3 + 0] * (1.0f - alpha_pct) + overlay.b * alpha_pct);
					row[x * 3 + 1] = static_cast<BYTE> (row[x * 3 + 1] * (1.0f - alpha_pct) + overlay.g * alpha_pct);
					row[x * 3 + 2] = static_cast<BYTE> (row[x * 3 + 2] * (1.0f - alpha_pct) + overlay.r * alpha_pct);
				}
			}
		}
		// --- 4. Pixel feedback/decay of the internal clip (Box blur attenuation) ---
		std::vector<PixelRGBA> next_buffer(width * height);
		auto get_pixel = [&](int x, int y) -> PixelRGBA
			{
				int cx = clamp(x, 0, width - 1);
				int cy = clamp(y, 0, height - 1);
				return internal_buffer[cy * width + cx];
			};
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				PixelRGBA top = get_pixel(x, y - 1);
				PixelRGBA bottom = get_pixel(x, y + 1);
				PixelRGBA left = get_pixel(x - 1, y);
				PixelRGBA right = get_pixel(x + 1, y);
				PixelRGBA top2 = get_pixel(x, y - 2);
				PixelRGBA bottom2 = get_pixel(x, y + 2);
				PixelRGBA left2 = get_pixel(x - 2, y);
				PixelRGBA right2 = get_pixel(x + 2, y);
				next_buffer[y * width + x].r = min((top.r + bottom.r + left.r + right.r) / 8 + (top2.r + bottom2.r + left2.r + right2.r) / 8, 255);
				next_buffer[y * width + x].g = min((top.g + bottom.g + left.g + right.g) / 8 + (top2.g + bottom2.g + left2.g + right2.g) / 8, 255);
				next_buffer[y * width + x].b = min((top.b + bottom.b + left.b + right.b) / 8 + (top2.b + bottom2.b + left2.b + right2.b) / 8, 255);
				next_buffer[y * width + x].a = min((top.a + bottom.a + left.a + right.a) / 10 + (top2.a + bottom2.a + left2.a + right2.a) / 12, 255);
			}
		}
		internal_buffer = std::move(next_buffer);
		return src;
	}
};

// Factory function for AviSynth+
AVSValue __stdcall Create_UltraVis(AVSValue args, void* user_data, IScriptEnvironment* env) { return new UltraVisualizer(args[0].AsClip(), env); }

// Plugin registration when loading the DLL
const AVS_Linkage* AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("UltraVis", "c", Create_UltraVis, 0);
	return "Ultra Visualizer";
}
