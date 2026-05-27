from __future__ import annotations

import argparse
import threading
import time
import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"


def poll_positions(client, stop_event, positions, hz=5.0):
    interval = 1.0 / hz

    while not stop_event.is_set():
        try:
            state = client.get_latest_state()
            pos = state.get("position") if state else None

            if pos and "x" in pos and "z" in pos:
                positions.append((time.time(), pos["x"], pos["z"]))

        except Exception:
            pass

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", default="clean_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minutes", type=float, default=4.0)

    args = parser.parse_args()

    duration = args.minutes * 60.0

    client = GameClient(SERVER_URL)

    session = client.create_session(
        mode="time_trial",
        player_name=f"clean_collector_{args.tag}",
        config={
            "seed": args.seed,
            "wind_enabled": False,
            "obstacles_enabled": False,
        },
    )

    print("\nOpen this URL in a NEW TAB and click inside the game:")
    print(session.get("browser_url"))

    print("\nGoal:")
    print("- Drive clean laps only")
    print("- Follow checkpoints")
    print("- Smooth steering")
    print("- Slow before turns")
    print("- Do NOT crash on purpose")
    print("- Do NOT do recovery training")
    print("- Do NOT zig-zag randomly")

    input("\nPress Enter when the bot is visible and browser is focused: ")

    client.connect_ws()
    time.sleep(0.5)

    positions = []
    stop_event = threading.Event()

    position_thread = threading.Thread(
        target=poll_positions,
        args=(client, stop_event, positions),
        daemon=True,
    )

    position_thread.start()

    client.start_recording(sample_rate=20)

    print(f"\nRecording {args.minutes:.1f} minutes. Drive now.\n")

    remaining = int(duration)

    while remaining > 0:
        print(f"... {remaining}s remaining")
        sleep_time = min(10, remaining)
        time.sleep(sleep_time)
        remaining -= sleep_time

    stop_event.set()

    info = client.stop_recording()
    print(f"\nStopped. Samples on server: {info.get('sample_count', '?')}")

    states, actions = client.get_recording_as_arrays()

    print(f"states shape  : {states.shape}")
    print(f"actions shape : {actions.shape}")

    positions_array = np.array(
        [(p[1], p[2]) for p in positions],
        dtype=np.float32,
    )

    print(f"positions shape: {positions_array.shape}")

    if states.ndim != 2 or states.shape[0] < 3000:
        raise RuntimeError(
            "Not enough samples saved. Keep browser focused and drive again."
        )

    output_name = f"data_{args.tag}.npz"

    np.savez(
        output_name,
        states=states.astype(np.float32),
        actions=actions.astype(np.float32),
        positions=positions_array,
        seed=np.array(args.seed, dtype=np.int32),
    )

    print(f"\nSaved {output_name}")


if __name__ == "__main__":
    main()