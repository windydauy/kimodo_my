"""Test event-aligned motion cropping: verify start/end positions match."""
import os
os.environ["HF_HOME"] = "/home/lina/humanoid/kimodo/huggingface"

from kimodo.training.g1_csv import load_g1_csv_motion

test_csv = "./datasets/g1/csv/210531/jump_and_land_heavy_001__A001.csv"
input_fps = 120

# Load full motion at 120fps
motion_data = load_g1_csv_motion(
    test_csv,
    source_coord_system="mujoco",
    root_euler_order="xyz",
    root_angle_unit="degrees",
    joint_angle_unit="degrees",
    root_position_unit="centimeters",
)
full_root = motion_data["root_positions"]  # [T, 3] at 120fps
print(f"Full clip: {full_root.shape[0]} frames at {input_fps}fps ({full_root.shape[0]/input_fps:.2f}s)")

# Check timeline events
from kimodo.training.timeline_annotations import TimelineAnnotationIndex
timeline = TimelineAnnotationIndex.from_jsonl("./datasets/SEED-Timeline-Annotations/timelines.jsonl")
rec = timeline.get_record(test_csv)

for i, event in enumerate(rec["events"][:5]):
    start_t = event["start_time"]
    end_t = event["end_time"]
    desc = event["description"]

    # Expected frame range at 120fps
    start_frame = int(start_t * input_fps)
    end_frame = min(int(end_t * input_fps), full_root.shape[0])

    # Crop manually
    cropped_root = full_root[start_frame:end_frame]

    # Verify: first frame of crop == full[start_frame]
    # Verify: last frame of crop == full[end_frame-1]
    first_match = (cropped_root[0] == full_root[start_frame]).all().item()
    last_match = (cropped_root[-1] == full_root[end_frame - 1]).all().item()

    print(f"\nEvent {i}: [{start_t:.1f}s - {end_t:.1f}s] {desc[:60]}...")
    print(f"  Frame range: [{start_frame} - {end_frame}] at {input_fps}fps")
    print(f"  Cropped frames: {cropped_root.shape[0]}")
    print(f"  After downsample (÷4): {cropped_root.shape[0]//4} frames at 30fps")
    print(f"  Root pos at start: {full_root[start_frame, :3].tolist()}")
    print(f"  Crop[0]:           {cropped_root[0, :3].tolist()}")
    print(f"  Start match: {first_match} ✓" if first_match else f"  Start match: {first_match} ✗")
    print(f"  Root pos at end:   {full_root[end_frame-1, :3].tolist()}")
    print(f"  Crop[-1]:          {cropped_root[-1, :3].tolist()}")
    print(f"  End match:   {last_match} ✓" if last_match else f"  End match:   {last_match} ✗")
