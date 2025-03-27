# env = create_snake_environment("./robotaxi/levels/8x8-blank.json", False, None, False, participant="participant_id_0")



from robotaxi.gameplay.wrappers import make_gymnasium_environment

env = make_gymnasium_environment("./robotaxi/levels/8x8-blank.json")
print(env.reset())
print(env.step(0))
#  (array([[6, 6, 6, 6, 6, 6, 6, 6],
#        [6, 0, 0, 0, 0, 0, 0, 6],
#        [6, 0, 0, 3, 4, 0, 0, 6],
#        [6, 0, 3, 0, 5, 0, 0, 6],
#        [6, 0, 0, 1, 0, 0, 0, 6],
#        [6, 0, 0, 1, 0, 0, 0, 6],
#        [6, 0, 0, 0, 0, 0, 0, 6],
#        [6, 6, 6, 6, 6, 6, 6, 6]]), 0, False, {})
