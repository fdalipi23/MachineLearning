# Instructor TODO — platform changes the project assumes

These are server-side / platform-side changes the project design assumes will be in place before students start. Listed in dependency order.

## Before students start iterating

<<<<<<< HEAD
- [x] **Honor the `seed` config in `create_session`.** `server/routes/session.ts` accepts both `terrain_seed` and `seed`, appends `&seed=N` to the browser URL, and `src/main.ts` feeds it into `TerrainGenerator` (Mulberry32 PRNG). `CheckpointSystem` uses a fixed `Math.sin(angle*3)*20` formula so checkpoint placement is identical across runs at any seed — same-seed reproducibility is what benchmarks need; all seeds share the same checkpoint layout by design.
- [x] **Verify the WS state broadcast includes `sensors.navigation.checkpoints_completed`.** Emitted in `src/main.ts` and consumed by `drive2win/eval.py`.
- [x] **Verify `state.position` is in the WS broadcast.** Emitted in `src/main.ts` and consumed by `drive2win/eval.py`.
- [x] **Add a `crashed` event or position-reset signal to the WS state.** `Agent.getRespawnCount()` is exposed as `sensors.navigation.respawn_count` in the WS broadcast. `eval.py` now prefers the counter-diff and only falls back to the old `dx² + dz² > 25` heuristic for older browser builds.

## Recording — needed for the path overlay and for any CNN iteration

- [x] **`RecordingSystem.captureSample()` should also record `position.x, position.z`.** Captured as `state.position_x` / `state.position_z` per sample; exposed via the new `client.get_recording_positions() -> np.ndarray` helper. Kept separate from the 12-feat BC training vector to preserve that contract.
- [x] **(Optional, for students who try a CNN.)** Pass `include_grid=True` to `client.start_recording(...)` to capture the 32×32×4 terrain grid per sample (`state.grid32`). Pull the data with `client.get_recording_with_grid()`, which returns `(states, actions, grid_stack)` with `grid_stack.shape == (N, 32, 32, 4)`. Bandwidth ≈ 5 MB per minute at 20 Hz.
=======
- [ ] **Honor the `seed` config in `create_session`.** `benchmark.py` and `01_collect.py` both pass `config.seed` so terrain layout + checkpoint placement are reproducible across students and across seeds. Verify this end-to-end (`SessionManager.ts` → `TimeTrialMode.ts` → `TerrainGenerator.ts`).
- [ ] **Verify the WS state broadcast includes `sensors.navigation.checkpoints_completed`.** Used by `eval.run_policy` to count progress. (See `wsClient.enableStateBroadcast` in `src/main.ts`; the structured-state object may need `nav.checkpoints_completed` exposed.)
- [ ] **Verify `state.position` is in the WS broadcast.** Both the in-run track plot AND the training-vs-test path overlay depend on it.
- [ ] **Add a `crashed` event or position-reset signal to the WS state.** The current heuristic in `eval.py` (`dx² + dz² > 25`) catches teleport-on-reset but is fragile. A first-class field is cleaner.

## Recording — needed for the path overlay and for any CNN iteration

- [ ] **`RecordingSystem.captureSample()` should also record `position.x, position.z`.** Currently it only captures the 12-feature state. Without this, students who want a high-Hz training path (instead of the low-Hz polled approximation in `01_collect.py`) can't get one. Suggested edit in `src/game/RecordingSystem.ts`:
  ```ts
  const pos = this.agent.getPosition();
  sample.state.x = pos.x;
  sample.state.z = pos.z;
  ```
  Then expose them through `client.get_recording_as_arrays()` so students can do `data["positions_full"]`.
- [ ] **(Optional, for students who try a CNN.)** Capture the 32×32 terrain grid per sample. Suggested:
  ```ts
  // src/game/RecordingSystem.ts captureSample()
  const grid = this.sensorSystem.getGrid32x32();   // already implemented
  sample.state.grid32 = grid.data;
  ```
  Plus a `client.get_recording_with_grid()` helper in `game_client.py`.
>>>>>>> 3ce6e615b6081f567cd14ab3c64c422361eb617c

## Before the tournament

- [ ] **Implement / verify the live tournament format the project assumes:**
  - 5 rounds × 5 minutes
  - terrain seed changes per round (use `seed = base + round_idx` or similar)
  - 3 rounds without obstacles, 2 with
  - pass / fail bar = ≥ 1 full lap completed in any round
  - ranking = total checkpoints across all 5 rounds
- [ ] **Capacity test the live arena for 20 simultaneous bots × 5 rounds.** Verify a disconnecting client doesn't take the room down. Verify scoring aggregates correctly across rounds.
- [ ] **Confirm `auth.ts` API-key flow works for 20+ concurrent clients hitting the same room.**
- [ ] **Decide and publish the deadline contract:** when does a student's agent need to be live and connected? What happens if they reconnect mid-round?

## Optional but valuable

- [ ] A `--fast` flag for `benchmark.py` that bumps `sim_speed` to 4× so 5 runs of 60 s wall-clock take ~75 s instead of 5 min. Useful during iteration.
- [ ] A separate **evaluation seed** the students cannot read — keeps the leaderboard fair if the published `seed=42` accidentally becomes a target for overfitting. (The 5-round terrain rotation already achieves much of this, but a hidden eval seed is a useful belt-and-suspenders.)
