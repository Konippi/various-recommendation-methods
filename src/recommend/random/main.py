from model import Random

TEST_SIZE = 0.2
USER_NUMS = 1000

if __name__ == "__main__":
    model = Random(USER_NUMS, TEST_SIZE)
    model.run()
