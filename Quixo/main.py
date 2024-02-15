from Game.ExtendedGame import ExtendedGame
import matplotlib.pyplot as plt
from players import RandomPlayer, HumanPlayer, MinMaxPlayer
from tqdm import tqdm

from matplotlib.ticker import MaxNLocator


def test_agent(num_games: int) -> None:
    count_0 = 0
    count_1 = 0

    for _ in range(num_games):
        game = ExtendedGame()
        player1 = MinMaxPlayer(game)
        player2 = RandomPlayer()

        winner = game.play(player1, player2)

        if winner == 0:
            count_0 += 1
        else:
            count_1 += 1
        print(f"{player1.name} win {count_0} matches")
        print(f"{player2.name} win {count_1} matches")


def test_agent_depths(num_games: int, max_depths: list[int]) -> None:
    results_when_first = []
    results_when_second = []

    for max_depth in tqdm(max_depths, desc="Testing depths"):
        wins_when_first = 0
        losses_when_first = 0
        wins_when_second = 0
        losses_when_second = 0

        for _ in tqdm(range(num_games), desc="Testing games"):
            # MinMaxPlayer starts first
            game = ExtendedGame()
            player1 = MinMaxPlayer(game, max_depth)
            player2 = RandomPlayer()
            winner = game.play(player1, player2)

            if winner == 0:
                wins_when_first += 1
            else:
                losses_when_first += 1

            # MinMaxPlayer starts second
            game = ExtendedGame()
            player1 = RandomPlayer()
            player2 = MinMaxPlayer(game, max_depth)
            winner = game.play(player1, player2)

            if winner == 1:
                wins_when_second += 1
            else:
                losses_when_second += 1

        results_when_first.append((max_depth, wins_when_first, losses_when_first))
        results_when_second.append((max_depth, wins_when_second, losses_when_second))

    return results_when_first, results_when_second


def plot_results(
    results: list[tuple[int, int, int]], filename: str, title: str
) -> None:
    depths, wins_0, wins_1 = zip(*results)
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(
        [d - width / 2 for d in depths],
        wins_0,
        width,
        label="MinMax Player",
        edgecolor="black",
    )
    ax.bar(
        [d + width / 2 for d in depths],
        wins_1,
        width,
        label="Random Player",
        edgecolor="black",
    )

    ax.set_title(title)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Number of Wins")
    ax.set_xticks(depths)
    ax.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Set y-axis to display only integers
    ax.legend()

    plt.savefig(filename)

    plt.show()


def play_against_ai() -> None:
    game = ExtendedGame()
    player1 = MinMaxPlayer(game)
    player2 = HumanPlayer()

    winner = game.play(player1, player2)

    if winner == 0:
        print(f"{player1.name} wins!")
    else:
        print(f"{player2.name} wins!")
    game.print()


if __name__ == "__main__":
    
    
    test_agent(100)
    
    # results_first, results_second = test_agent_depths(
    #     100, [10]
    # )
    # plot_results(results_first, "results_first.png", "MinMax Player starts first")
    # plot_results(results_second, "results_second.png", "MinMax Player starts second")
    
    # play_against_ai()
