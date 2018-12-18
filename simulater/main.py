from simulater import Environment as en
from view import ArrayView as av

env=en.Environment()
env.step(0)
env.step(1)
env.step(0)

env.step(1)
env.step(2)
env.step(3)
#env.step(0)


av.visualize_space(env.SPACE.RESULTS)
