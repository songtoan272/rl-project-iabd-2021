import random
import json
import operator

import to_do.dynamic_programming as dynamic_programming
import to_do.monte_carlo_methods as monte_carlo_methods
import to_do.temporal_difference_learning as temporal_difference_learning
import to_do.chess as chess


def export_to_json(pi, file_name: str):
    action_by_state = {int(str(k)): int(max(v.items(), key=operator.itemgetter(1))[0]) if type(v) is dict else int(str(v)) for k, v in pi.items()}

    pi_json = json.dumps(action_by_state)
    f = open('./models/' + file_name + '.json', 'w')
    f.write(pi_json)
    f.close()


if __name__ == "__main__":
    print('\n\n\nDynamic\n\n\n')
    #dynamic_programming.demo()
    print('\n\n\nmonte_carlo\n\n\n')
    #monte_carlo_methods.demo()
    print('\n\n\ntemporal\n\n\n')
    temporal_difference_learning.demo()
