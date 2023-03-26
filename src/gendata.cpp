// this file is used to generate game data by playing with itself.

#include <random>

#include "game.h"
#include "strategy.h"

// 每个样本随机的次数
const int SAMPLE_TIMES = 200;

std::mt19937 rng(std::random_device{}());

int main() {
  freopen("data.txt", "w", stdout);
  std::string actions;
  float probs[28];
  StraightStrategy straight;
  DeepStrategy deep;
  Deep2Strategy deep2;

  for (;;) {
    for (int steps = 1; steps <= 10; steps++) {
      // generate games with turn = steps
      Game g;
      while (g.turn < steps) {
        // first step is chance.
        g.get_chances(actions, probs);
        std::discrete_distribution<int> dist(probs, probs + actions.size());
        int chanceIndex = dist(rng);
        char chanceAction = actions[chanceIndex];
        g.step(chanceAction);
        actions.clear();

        // second step is action.
        char action;
        // 为了数据更丰富，有1%的几率不选择最优步而是走随机步
        if (rand() % 100 == 0) {
          g.get_actions(actions);
          std::uniform_int_distribution<int> dist(0, actions.size() - 1);
          int index = dist(rng);
          action = actions[index];
          actions.clear();
        } else {
          action = straight.getAction(g);
        }
        g.step(action);
      }

      for (int i = 0; i < 20; i++) {
        printf("%d ", g.board[i]);
      }
      int totalTimes = 0;
      double totalScore = 0;

      while (totalTimes < SAMPLE_TIMES) {
        Game game = g;

        while (!game.is_ended()) {
          // first step is chance.
          game.get_chances(actions, probs);
          std::discrete_distribution<int> dist(probs, probs + actions.size());
          int chanceIndex = dist(rng);
          char chanceAction = actions[chanceIndex];
          game.step(chanceAction);
          actions.clear();

          // second step is action.
          char action;
          if (game.turn < 15) {
            action = straight.getAction(game);
          } else {
            action = deep2.getAction(game);
          }
          game.step(action);
        }

        int finalScore = game.get_score();
        totalScore += finalScore;
        totalTimes++;
      }

      double mean = totalScore / totalTimes;
      printf("%.4lf\n", mean);
      fflush(stdout);
    }
  }

  return 0;
}
