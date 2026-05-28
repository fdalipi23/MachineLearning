from drive2win.my_agent import make_policy
from game_client import RoomBot

policy = make_policy("nav_v19_clean.npz")

bot = RoomBot(
    "https://ml.ferit.tech",
    room="tournament7",
    name="Fiorela"
)

standings = bot.run(policy, hz=20.0)

print(standings)
