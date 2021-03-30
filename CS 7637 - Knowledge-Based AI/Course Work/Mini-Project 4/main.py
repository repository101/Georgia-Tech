from MonsterClassificationAgent import MonsterClassificationAgent

def test():
    tests = load_tests()
    #This will run your code against the first four known test cases.
    test_agent = MonsterClassificationAgent()

    known_positive_1 = {'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True}
    known_positive_2 = {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': False}
    known_positive_3 = {'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': True}
    known_positive_4 = {'size': 'large', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 3, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True}
    known_positive_5 = {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False}
    known_negative_1 = {'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True}
    known_negative_2 = {'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 8, 'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has_gills': False, 'has-tail': False}
    known_negative_3 = {'size': 'medium', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 6, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has_gills': False, 'has-tail': False}
    known_negative_4 = {'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 6, 'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False}
    known_negative_5 = {'size': 'medium', 'color': 'purple', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has_gills': True, 'has-tail': False}

    monster_list = [(known_positive_1, True),
                    (known_positive_2, True),
                    (known_positive_3, True),
                    (known_positive_4, True),
                    (known_positive_5, True),
                    (known_negative_1, False),
                    (known_negative_2, False),
                    (known_negative_3, False),
                    (known_negative_4, False),
                    (known_negative_5, False)]
    
    # True
    new_monster_1 = {'size': 'large', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 3, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True}
    # False
    new_monster_2 = {'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 8, 'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has_gills': False, 'has-tail': False}
    # False
    new_monster_3 = {'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 3, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False}
    # True
    new_monster_4 = {'size': 'small', 'color': 'black', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4, 'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False}

    print(test_agent.solve(tests, new_monster_1))
    print(test_agent.solve(monster_list, new_monster_2))
    print(test_agent.solve(monster_list, new_monster_3))
    print(test_agent.solve(monster_list, new_monster_4))


def load_tests():
    a = [({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
       'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True},
      True), ({'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
               'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
               'has-tail': False}, True), (
     {'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': True},
     True), ({'size': 'large', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 3,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, True), (
     {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False},
     True), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, False), (
     {'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 8,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 6,
               'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has_gills': False,
               'has-tail': False}, False), (
     {'size': 'medium', 'color': 'purple', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has_gills': True, 'has-tail': False},
     False)]
    a_t = [True, True, True, True, True, False, False, False, False, False]
    
    b = [({'size': 'medium', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 7,
       'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
      True), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'large', 'color': 'red', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 7, 'arm-count': 5,
              'eye-count': 2, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, False), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 8, 'arm-count': 0,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'gray', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 5,
              'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'small', 'color': 'red', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 7, 'arm-count': 0,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'small', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 8, 'arm-count': 5,
              'eye-count': 8, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'huge', 'color': 'green', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'black', 'covering': 'skin', 'foot-type': 'foot', 'leg-count': 5, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'gray', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 7,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 4,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 0, 'arm-count': 0,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'green', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True)]
    b_t = [True, True, False, False, True, True, False, True, True, False, False, False, True, True, False, False, False,
     True]
    
    c = [({'size': 'medium', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 7,
       'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
      True), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'large', 'color': 'red', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 7, 'arm-count': 5,
              'eye-count': 2, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, False), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 8, 'arm-count': 0,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'gray', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 5,
              'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'small', 'color': 'red', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 7, 'arm-count': 0,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'small', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 8, 'arm-count': 5,
              'eye-count': 8, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'huge', 'color': 'green', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'black', 'covering': 'skin', 'foot-type': 'foot', 'leg-count': 5, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'gray', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 7,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 4,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 0, 'arm-count': 0,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'green', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True)]
    c_t = [True, True, False, False, True, True, False, True, True, False, False, False, True, True, False, False, False,
     True]
    
    d = [({'size': 'huge', 'color': 'red', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 7,
       'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
      False), (
     {'size': 'large', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 2, 'arm-count': 1,
      'eye-count': 4, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 2,
              'eye-count': 4, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'white', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 0,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 1,
               'eye-count': 2, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 2,
      'eye-count': 6, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 7,
      'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'hoof', 'leg-count': 0, 'arm-count': 0,
      'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 1,
               'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'medium', 'color': 'gray', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 0,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'white', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 2,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 1,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'huge', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 3, 'arm-count': 2,
              'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 2,
      'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 3, 'arm-count': 1,
              'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 7,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 0,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 4,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
               'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 4, 'arm-count': 1,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'gray', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 4, 'arm-count': 2,
              'eye-count': 6, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'white', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 2,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'medium', 'color': 'blue', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 2,
      'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 6,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 1,
               'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
               'has-tail': True}, False), (
     {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 8,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 1,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 1,
              'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': False}, True)]
    d_t = [False, True, True, False, True, True, False, False, True, False, False, True, True, True, True, False, True, False,
     False, False, True, True, True, True, False, False, False, False, False, False, True, True]
    
    e = [({'size': 'large', 'color': 'white', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 4,
       'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
      False), (
     {'size': 'large', 'color': 'brown', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 8,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'red', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 5,
      'eye-count': 1, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'green', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 5, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'large', 'color': 'black', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'gray', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 3,
              'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 1,
      'eye-count': 1, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 8,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 3,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'gray', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 4,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'small', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True)]
    e_t = [False, False, True, False, True, False, False, True, True, False, True, True, False, True, True, False, True, True,
     False, False, True, True, False, False, False, False, True, True]
    
    f = [({'size': 'huge', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 3,
       'eye-count': 1, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
      False), (
     {'size': 'tiny', 'color': 'red', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'green', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 5, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'orange', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 1,
      'eye-count': 6, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 3,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'red', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 1,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'small', 'color': 'black', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 4, 'arm-count': 6,
              'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, False), (
     {'size': 'medium', 'color': 'white', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'red', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 2,
               'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': True}, True), (
     {'size': 'medium', 'color': 'green', 'covering': 'skin', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 1,
              'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
              'has-tail': False}, False), (
     {'size': 'medium', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'orange', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 8, 'arm-count': 2,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 3,
              'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 1,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'tiny', 'color': 'red', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 2,
               'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': True}, True)]
    f_t = [False, True, True, True, False, False, True, False, False, True, True, False, False, False, True, True, False,
     True]
    
    g = [({'size': 'medium', 'color': 'orange', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 2,
       'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
      True), (
     {'size': 'medium', 'color': 'orange', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'tiny', 'color': 'red', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 6, 'arm-count': 7,
              'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False,
              'has-tail': False}, False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'red', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 8,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 3,
      'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'large', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'huge', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 4,
              'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': False,
              'has-tail': False}, True), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 0,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False)]
    g_t = [True, False, True, False, False, False, True, True, True, False]
    
    h = [({'size': 'tiny', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 6,
       'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
      True), (
     {'size': 'small', 'color': 'brown', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 7,
      'eye-count': 1, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'medium', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 7,
              'eye-count': 8, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, False), (
     {'size': 'large', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 2,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 8,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'blue', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 5,
              'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'small', 'color': 'black', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 5,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'white', 'covering': 'skin', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 3,
              'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'small', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'black', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 7,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 6, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 6, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'huge', 'color': 'brown', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 8,
              'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'small', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 1, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'green', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 6, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'black', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), ({'size': 'small', 'color': 'brown', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 5,
              'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), ({'size': 'tiny', 'color': 'blue', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 0,
              'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, False), (
     {'size': 'small', 'color': 'black', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'green', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 6, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'green', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'black', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 7,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 2,
      'eye-count': 4, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False)]
    h_t = [True, True, True, False, False, False, True, True, False, True, False, False, True, False, True, False, True,
     False, True, False, False, False, False, True, True, True, False, True, True, True, False, True, False, False]
    
    i = [({'size': 'large', 'color': 'black', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 3,
       'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
      False), (
     {'size': 'large', 'color': 'green', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 7,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'red', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'brown', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 8,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 5, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'gray', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 8, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'black', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 6, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'gray', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 5, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 8,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'large', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 0,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'green', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 7,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'brown', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 7,
      'eye-count': 5, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'brown', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 5,
               'eye-count': 1, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False,
               'has-tail': False}, False), (
     {'size': 'huge', 'color': 'black', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'green', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 3,
      'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'large', 'color': 'red', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 7,
               'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': True}, False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'red', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 6, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 6,
              'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, False)]
    i_t = [False, False, True, True, True, False, False, True, True, False, True, True, True, False, True, True, False, False,
     False, False, False, True, True, False]
    
    j = [({'size': 'tiny', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 6,
       'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
      True), (
     {'size': 'small', 'color': 'brown', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 7,
      'eye-count': 1, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'medium', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 7,
              'eye-count': 8, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, False), (
     {'size': 'large', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 2,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 8,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'blue', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 5,
              'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'small', 'color': 'black', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 5,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'white', 'covering': 'skin', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 4,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 3,
              'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'small', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'black', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 7,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 6, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 6, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), ({'size': 'huge', 'color': 'brown', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 8,
              'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'small', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 1, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'green', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 6, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'black', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), ({'size': 'small', 'color': 'brown', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 5,
              'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), ({'size': 'tiny', 'color': 'blue', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 0,
              'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, False), (
     {'size': 'small', 'color': 'black', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'green', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 6, 'arm-count': 6,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'green', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'black', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 7,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 2,
      'eye-count': 4, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False)]
    j_t = [True, True, True, False, False, False, True, True, False, True, False, False, True, False, True, False, True,
     False, True, False, False, False, False, True, True, True, False, True, True, True, False, True, False, False]
    
    k = [({'size': 'huge', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 6, 'arm-count': 3,
       'eye-count': 1, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
      False), (
     {'size': 'tiny', 'color': 'red', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'green', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 5, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'orange', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 1,
      'eye-count': 6, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 3,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'red', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 1,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'small', 'color': 'black', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 4, 'arm-count': 6,
              'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
              'has-tail': True}, False), (
     {'size': 'medium', 'color': 'white', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'red', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 2,
               'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': True}, True), (
     {'size': 'medium', 'color': 'green', 'covering': 'skin', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 1,
              'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
              'has-tail': False}, False), (
     {'size': 'medium', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'orange', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 8, 'arm-count': 2,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 3,
              'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 1,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'tiny', 'color': 'red', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 2,
               'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': True}, True)]
    k_t = [False, True, True, True, False, False, True, False, False, True, True, False, False, False, True, True, False,
     True]
    
    l = [({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
       'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True},
      True), ({'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
               'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
               'has-tail': False}, True), (
     {'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': True},
     True), ({'size': 'large', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 3,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, True), (
     {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False},
     True), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, False), (
     {'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 8,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 6,
               'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has_gills': False,
               'has-tail': False}, False), (
     {'size': 'medium', 'color': 'purple', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has_gills': True, 'has-tail': False},
     False)]
    l_t = [True, True, True, True, True, False, False, False, False, False]
    
    m = [({'size': 'large', 'color': 'white', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 4,
       'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
      False), (
     {'size': 'large', 'color': 'brown', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 8,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'red', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 5,
      'eye-count': 1, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'green', 'covering': 'skin', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 5, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'large', 'color': 'black', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'tiny', 'color': 'gray', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 3,
              'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 1,
      'eye-count': 1, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 8,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 3,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'gray', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 4,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'small', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True)]
    m_t = [False, False, True, False, True, False, False, True, True, False, True, True, False, True, True, False, True, True,
     False, False, True, True, False, False, False, False, True, True]
    
    n = [({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
       'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True},
      True), ({'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
               'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
               'has-tail': False}, True), (
     {'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': True},
     True), ({'size': 'large', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 3,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, True), (
     {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False},
     True), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, False), (
     {'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 8,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 6,
               'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has_gills': False,
               'has-tail': False}, False), (
     {'size': 'medium', 'color': 'purple', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has_gills': True, 'has-tail': False},
     False)]
    n_t = [True, True, True, True, True, False, False, False, False, False]
    
    o = [({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
       'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True, 'has-tail': True},
      True), ({'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
               'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
               'has-tail': False}, True), (
     {'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': True},
     True), ({'size': 'large', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 3,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, True), (
     {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': False, 'has-tail': False},
     True), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 4,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has_gills': True,
              'has-tail': True}, False), (
     {'size': 'tiny', 'color': 'red', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 8,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'gray', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has_gills': False, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'black', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 6,
               'eye-count': 2, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has_gills': False,
               'has-tail': False}, False), (
     {'size': 'medium', 'color': 'purple', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 4,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has_gills': True, 'has-tail': False},
     False)]
    o_t = [True, True, True, True, True, False, False, False, False, False]
    
    p = [({'size': 'large', 'color': 'brown', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 7,
       'eye-count': 1, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
      False), (
     {'size': 'huge', 'color': 'green', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'green', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 0,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'black', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 4,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'small', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 6,
               'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
               'has-tail': False}, True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 3,
      'eye-count': 2, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'blue', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'green', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 2, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'large', 'color': 'red', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 6, 'arm-count': 7,
              'eye-count': 0, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 4,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
              'has-tail': False}, True), (
     {'size': 'large', 'color': 'orange', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'small', 'color': 'red', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 4,
               'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'small', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'red', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 3,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'tiny', 'color': 'red', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 8, 'arm-count': 4,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'medium', 'color': 'red', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 4,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False,
              'has-tail': False}, True), (
     {'size': 'small', 'color': 'green', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 1,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'white', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 3,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'huge', 'color': 'red', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 0,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'green', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'large', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 4,
      'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 4,
               'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
               'has-tail': False}, True)]
    p_t = [False, True, False, True, False, False, True, False, False, False, True, False, True, True, False, True, True,
     True, True, True, False, True, True, False, False, False, True, False, False, True]
    
    q = [({'size': 'medium', 'color': 'orange', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 2,
       'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
      True), (
     {'size': 'medium', 'color': 'orange', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 3,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'tiny', 'color': 'red', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 6, 'arm-count': 7,
              'eye-count': 8, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': False,
              'has-tail': False}, False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'red', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 8,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'brown', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 3,
      'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'large', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'huge', 'color': 'blue', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 4,
              'eye-count': 5, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': False,
              'has-tail': False}, True), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 0,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False)]
    q_t = [True, False, True, False, False, False, True, True, True, False]
    
    r = [({'size': 'large', 'color': 'brown', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 7,
       'eye-count': 1, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
      False), (
     {'size': 'huge', 'color': 'green', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'green', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 0,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'black', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 4,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'green', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 0, 'arm-count': 2,
      'eye-count': 3, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'small', 'color': 'white', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 6,
               'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
               'has-tail': False}, True), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 6, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'red', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 3,
      'eye-count': 2, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'blue', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 4,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'green', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 2, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'large', 'color': 'red', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 6, 'arm-count': 7,
              'eye-count': 0, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 4,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False,
              'has-tail': False}, True), (
     {'size': 'large', 'color': 'orange', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'small', 'color': 'red', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 4,
               'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'small', 'color': 'green', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'red', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 3,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'tiny', 'color': 'red', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 8, 'arm-count': 4,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'medium', 'color': 'red', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 4,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False,
              'has-tail': False}, True), (
     {'size': 'small', 'color': 'green', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 1,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'white', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'white', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 3,
              'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'huge', 'color': 'red', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'feathers', 'foot-type': 'hoof', 'leg-count': 6, 'arm-count': 0,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 2, 'arm-count': 2,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'green', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 2, 'arm-count': 6,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     True), (
     {'size': 'large', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 4,
      'eye-count': 4, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'medium', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'huge', 'color': 'red', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 0, 'arm-count': 4,
               'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
               'has-tail': False}, True)]
    r_t = [False, True, False, True, False, False, True, False, False, False, True, False, True, True, False, True, True,
     True, True, True, False, True, True, False, False, False, True, False, False, True]
    
    s = [({'size': 'large', 'color': 'black', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 8, 'arm-count': 3,
       'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
      False), (
     {'size': 'large', 'color': 'green', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 7,
      'eye-count': 8, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'red', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'brown', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'black', 'covering': 'skin', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 8,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 5, 'arm-count': 7,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'gray', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 8, 'arm-count': 6,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'medium', 'color': 'black', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 3,
      'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'medium', 'color': 'brown', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 6, 'arm-count': 8,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'tiny', 'color': 'gray', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 5, 'arm-count': 6,
      'eye-count': 2, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'medium', 'color': 'white', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 8, 'arm-count': 8,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'large', 'color': 'brown', 'covering': 'feathers', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 0,
      'eye-count': 5, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'green', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 4, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 7,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'brown', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 1, 'arm-count': 7,
      'eye-count': 5, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'brown', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 5,
               'eye-count': 1, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False,
               'has-tail': False}, False), (
     {'size': 'huge', 'color': 'black', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'green', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 3,
      'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'large', 'color': 'red', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 7,
               'eye-count': 3, 'horn-count': 2, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': True}, False), (
     {'size': 'huge', 'color': 'gray', 'covering': 'scales', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'small', 'color': 'red', 'covering': 'scales', 'foot-type': 'foot', 'leg-count': 6, 'arm-count': 5,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 1, 'arm-count': 6,
              'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, False)]
    s_t = [False, False, True, True, True, False, False, True, True, False, True, True, True, False, True, True, False, False,
     False, False, False, True, True, False]
    
    t = [({'size': 'huge', 'color': 'red', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 7,
       'eye-count': 7, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
      False), (
     {'size': 'large', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 2, 'arm-count': 1,
      'eye-count': 4, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 2,
              'eye-count': 4, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'white', 'covering': 'skin', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 0,
      'eye-count': 0, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), ({'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 3, 'arm-count': 1,
               'eye-count': 2, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 2,
      'eye-count': 6, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     True), (
     {'size': 'huge', 'color': 'blue', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 4, 'arm-count': 7,
      'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'hoof', 'leg-count': 0, 'arm-count': 0,
      'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 2, 'arm-count': 1,
               'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'medium', 'color': 'gray', 'covering': 'feathers', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 0,
      'eye-count': 2, 'horn-count': 2, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'white', 'covering': 'feathers', 'foot-type': 'foot', 'leg-count': 2, 'arm-count': 2,
      'eye-count': 6, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'huge', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 3, 'arm-count': 1,
      'eye-count': 2, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'huge', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 3, 'arm-count': 2,
              'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 3, 'arm-count': 2,
      'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     True), ({'size': 'large', 'color': 'gray', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 3, 'arm-count': 1,
              'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, True), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'scales', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 7,
      'eye-count': 0, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': True},
     False), (
     {'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 4, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), (
     {'size': 'small', 'color': 'yellow', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 5, 'arm-count': 5,
      'eye-count': 6, 'horn-count': 0, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'tiny', 'color': 'orange', 'covering': 'skin', 'foot-type': 'none', 'leg-count': 0, 'arm-count': 0,
      'eye-count': 8, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'tiny', 'color': 'brown', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 0, 'arm-count': 4,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     False), ({'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 2,
               'eye-count': 3, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
               'has-tail': False}, True), (
     {'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 4, 'arm-count': 1,
      'eye-count': 7, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'gray', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 4, 'arm-count': 2,
              'eye-count': 6, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True,
              'has-tail': False}, True), (
     {'size': 'large', 'color': 'blue', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 1, 'arm-count': 2,
      'eye-count': 7, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'small', 'color': 'white', 'covering': 'fur', 'foot-type': 'none', 'leg-count': 1, 'arm-count': 2,
              'eye-count': 2, 'horn-count': 0, 'lays-eggs': True, 'has-wings': True, 'has-gills': True,
              'has-tail': True}, False), (
     {'size': 'medium', 'color': 'blue', 'covering': 'fur', 'foot-type': 'talon', 'leg-count': 7, 'arm-count': 2,
      'eye-count': 6, 'horn-count': 2, 'lays-eggs': True, 'has-wings': True, 'has-gills': False, 'has-tail': True},
     False), (
     {'size': 'small', 'color': 'yellow', 'covering': 'scales', 'foot-type': 'talon', 'leg-count': 1, 'arm-count': 5,
      'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': True, 'has-gills': True, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'orange', 'covering': 'feathers', 'foot-type': 'talon', 'leg-count': 3, 'arm-count': 6,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': True, 'has-wings': False, 'has-gills': False, 'has-tail': False},
     False), ({'size': 'huge', 'color': 'blue', 'covering': 'fur', 'foot-type': 'foot', 'leg-count': 7, 'arm-count': 1,
               'eye-count': 8, 'horn-count': 2, 'lays-eggs': True, 'has-wings': False, 'has-gills': True,
               'has-tail': True}, False), (
     {'size': 'large', 'color': 'white', 'covering': 'fur', 'foot-type': 'hoof', 'leg-count': 5, 'arm-count': 8,
      'eye-count': 0, 'horn-count': 1, 'lays-eggs': False, 'has-wings': True, 'has-gills': False, 'has-tail': False},
     False), (
     {'size': 'large', 'color': 'yellow', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 5, 'arm-count': 1,
      'eye-count': 3, 'horn-count': 1, 'lays-eggs': False, 'has-wings': False, 'has-gills': True, 'has-tail': False},
     True), ({'size': 'huge', 'color': 'gray', 'covering': 'fur', 'foot-type': 'paw', 'leg-count': 4, 'arm-count': 1,
              'eye-count': 5, 'horn-count': 0, 'lays-eggs': False, 'has-wings': False, 'has-gills': True,
              'has-tail': False}, True)]
    t_t = [False, True, True, False, True, True, False, False, True, False, False, True, True, True, True, False, True, False,
     False, False, True, True, True, True, False, False, False, False, False, False, True, True]

    samples = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t]
    results = []
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            results.append(samples[i][j])
    return results


if __name__ == "__main__":
    test()