from drive2win.temporal import make_policy

def main():
    policy = make_policy("nav_v7.pt")
    
    # Mock state matching the WebSocket payload structure
    mock_state = {
        "sensors": {
            "speed": 0.0,
            "heading_error": 0.0,
            "checkpoint_distance": 30.0,
            "rays": [50.0] * 8,
            "ground_friction": 1.0
        }
    }
    
    # Try calling it a few times to check buffer stacking
    for i in range(5):
        action = policy(mock_state)
        print(f"Call {i+1} action:", action)

if __name__ == "__main__":
    main()
