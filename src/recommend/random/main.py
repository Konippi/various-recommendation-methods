from model import Random
from params import TEST_SIZE, USER_NUMS

if __name__ == "__main__":
    model = Random(USER_NUMS, TEST_SIZE)
    model.run()
