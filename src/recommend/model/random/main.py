from model import Random

USER_NUMS: int = 1000
TEST_SIZE: float = 0.2

if __name__ == "__main__":
    model = Random(USER_NUMS, TEST_SIZE)
    model.run()
