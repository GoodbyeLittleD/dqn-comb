// this file is used to benchmark by playing with itself.

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
  int totalTimes = 0;
  double totalScore = 0;
  double totalSquare = 0;

  for (;;) {
    Game g;
    while (!g.is_ended()) {
      // first step is chance.
      g.get_chances(actions, probs);
      std::discrete_distribution<int> dist(probs, probs + actions.size());
      int chanceIndex = dist(rng);
      char chanceAction = actions[chanceIndex];
      g.step(chanceAction);
      actions.clear();
      //   if (chanceAction >= 55) {
      //     printf("turn %d, square = *\n");
      //   } else {
      //     printf("turn %d, square = %d %d %d\n", g.turn, get_left(chanceAction),
      //            get_mid(chanceAction), get_right(chanceAction));
      //   }

      // second step is action.
      char action;
      // if (g.turn < 15) {
      //   action = straight.getAction(g);
      // } else {
      //   action = deep2.getAction(g);
      // }
      action = straight.getAction(g);
      g.step(action);
      // printf("action = %d\n\n", (int)action);
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