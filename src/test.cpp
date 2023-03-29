// this file is used to generate game data by playing with itself.

#include <random>

#include "game.h"
#include "strategy.h"

std::mt19937 rng(std::random_device{}());

int main() {
  std::string actions;
  float probs[28];
  StraightStrategy straight;
  DeepStrategy deep;
  Deep2Strategy deep2;
  RandomStrategy rando;
  // NetStrategy net;
  DeepNetStrategy net;
  int totalTimes = 0;
  double totalScore = 0;
  double totalSquare = 0;

  for (;;) {
    Game g;
    while (!g.is_ended()) {
      // auto currentScore = getExpScore(g);
      // printf("current score = %lf\n", currentScore);

      // first step is chance.
      g.get_chances(actions, probs);
      std::discrete_distribution<int> dist(probs, probs + actions.size());
      int chanceIndex = dist(rng);
      char chanceAction;
      chanceAction = actions[chanceIndex];

      // int a, b, c;
      // printf("Input card (a,b,c): ");
      // scanf("%d%d%d", &a, &b, &c);
      // chanceAction = (a == 3   ? 0
      //                 : a == 4 ? 1
      //                          : 2) *
      //                    18 +
      //                (b == 1   ? 0
      //                 : b == 5 ? 1
      //                          : 2) *
      //                    6 +
      //                (c == 2   ? 0
      //                 : c == 6 ? 1
      //                          : 2) *
      //                    2;
      // if (a == 10 && b == 10 && c == 10) {
      //   chanceAction = 54;
      // }
      // if (!(g.flag & (1ull << chanceAction))) {
      //   chanceAction++;
      // }

      // if (chanceAction >= 54) {
      //   printf("turn %d, square = *\n");
      // } else {
      //   printf("turn %d, square = %d %d %d\n", g.turn, get_left(chanceAction),
      //          get_mid(chanceAction), get_right(chanceAction));
      // }
      g.step(chanceAction);
      actions.clear();

      // second step is action.
      char action;
      if (g.turn < 10) {
        action = net.getAction(g);
        // action = rando.getAction(g);
      } else {
        action = deep2.getAction(g);
      }
      g.step(action);
      // printf("best_action = %d\n\n", (int)action);
    }
    int finalScore = g.get_score();
    totalScore += finalScore;
    totalSquare += finalScore * finalScore;
    totalTimes++;
    double mean = totalScore / totalTimes;
    double sd = sqrt(totalSquare / totalTimes - mean * mean);
    double margin = sd / sqrt(totalTimes) * 2.58;
    printf(
        "final score: %d\naverage score = %lf  standard deviation = %lf  total times = "
        "%d\nconfidence interval(99%) = [%lf, %lf]\n",
        finalScore, mean, sd, totalTimes, mean - margin, mean + margin);
  }

  return 0;
}