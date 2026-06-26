import numpy as np

def index_to_frame(videos, i):
    try:
        v_idx, f_idx = i
        # Access the frame
        frame = videos[v_idx][f_idx]
        return frame

    except Exception as e:
        print("  Raw index:", repr(i))
        print(" Len videos:", len(videos))
        print(" Len video i:", len(videos[i[0]]))
        assert False

def past_frames(video_index, t, num_frames):
    N_available = min(t+1, num_frames)
    if N_available == 1:
        return [(video_index, 0)]
    step = t / num_frames
    indices = [round(k * step) for k in range(N_available)]
    return [(video_index, i) for i in indices]

def past_frames_random(video_index, t, num_frames):
    N_available = min(t+1, num_frames)
    if N_available == 1:
        return [(video_index, 0)]
    step = t / num_frames
    steps = np.cumsum(np.random.rand(num_frames) * step * 0.5 + 0.5 * step) - 0.5 * step
    indices = [round(k) for k in steps]
    return [(video_index, i) for i in indices]

def all_past_frames(video_index, t, num_frames):
    return [(video_index, i) for i in range(t+1)]

def past_frames_single_video(t, num_frames):
    N_available = min(t+1, num_frames)
    if N_available == 1:
        return [0]
    step = t / num_frames
    indices = [round(k * step) for k in range(N_available)]
    return indices

def all_past_frames_single_video(t):
    return [i for i in range(t+1)]