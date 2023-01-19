import random

user_action = input("你要出：石头(1)，剪刀(2)，布(3)\n")
possible_actions = ["石头", "剪刀", "布"]
computer_action = random.choice(possible_actions)
print(f"\n你出的是 {possible_actions[int(user_action)-1]}, 电脑出的是{computer_action}.\n")

if user_action == computer_action:
    print(f"都出的是 {user_action}. 本次平局!")
elif user_action == "1":
    if computer_action == "剪刀":
        print("石头砸剪刀! 你赢了!")
    else:
        print("布包石头! 你输了.")
elif user_action == "3":
    if computer_action == "石头":
        print("布包石头! 你赢了!")
    else:
        print("剪刀剪布! 你输了.")
elif user_action == "2":
    if computer_action == "布":
        print("剪刀剪布! 你赢了!")
    else:
        print("石头砸剪刀! 你输了.")