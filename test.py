


def run_seq(start,step,until_equal):
    x = 0
    iter_ = 0
    while x != until_equal:
        x += (start - step * iter_)
        iter_ += 1
    print(iter_)
    print((800 - 25 * 8)*8)


if __name__ == "__main__":
    run_seq(800,25,5700)