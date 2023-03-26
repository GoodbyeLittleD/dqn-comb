#include <cmath>
#include <iostream>

#include "game.h"

static const int NUM[3][3] = {
    {3, 4, 8},
    {1, 5, 9},
    {2, 6, 7},
};

static const int NUM4[3][4] = {
    {8, 4, 3, 0},
    {9, 5, 1, 0},
    {7, 6, 2, 0},
};

static const int LINES[3][5][5] = {
    {
        {8, 13, 17},
        {4, 9, 14, 18},
        {1, 5, 10, 15, 19},
        {2, 6, 11, 16},
        {3, 7, 12},
    },
    {
        {1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19},
    },
    {
        {1, 4, 8},
        {2, 5, 9, 13},
        {3, 6, 10, 14, 17},
        {7, 11, 15, 18},
        {12, 16, 19},
    },
};

enum Status {
  EMPTY,
  PARTIAL,
  FULL,
  BROKEN,
};

static const double vars[11] = {1.00, 0.721, 0.3993,  0.1947,  0.069, 0.0312,
                                0.75, 0.03,  0.08465, 0.08164, 18};

double getExpScore(const Game& g) {
  int cardList[20][3] = {0};
  for (int i = 0; i < 20; i++) {
    if (g.board[i] >= 55) {
      cardList[i][0] = cardList[i][1] = cardList[i][2] = 10;
    } else if (g.board[i] >= 0) {
      cardList[i][0] = get_left(g.board[i]);
      cardList[i][1] = get_mid(g.board[i]);
      cardList[i][2] = get_right(g.board[i]);
    }
  }

  printf("calling getExpScore. cardlist: [%d %d %d] [%d %d %d] [%d %d %d] ...\n", cardList[0][0],
         cardList[0][1], cardList[0][2], cardList[1][0], cardList[1][1], cardList[1][2],
         cardList[2][0], cardList[2][1], cardList[2][2]);

  double ret = 0, waiting[10] = {0}, decide[20][2] = {0};
  int blockCount = 0, lastNum = 0, lastScore = 0, desired[10] = {0}, needs[10] = {0};
  Status status;
  int length, score, num, filled;
  double scale, times;

  for (int i = 0; i < 20; i++)
    if (cardList[i][0]) blockCount++;

  if (cardList[0][0] == 0) {
    ret += vars[10];
  } else {
    ret += cardList[0][0] + cardList[0][1] + cardList[0][2];
  }

  printf("after first step, ret = %lf\n", ret);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      length = 5 - abs(j - 2);
      score = 0;
      num = 0;
      filled = 0;

      bool flag = true;
      for (int k = 0; k < length && flag; k++) {
        int now = cardList[LINES[i][j][k]][i];
        if (now != 0) filled++;
        if (now != 0 && now != 10) {
          if (num != 0 && num != now) {
            status = BROKEN;
            score = 0;
            flag = false;
          } else {
            num = now;
          }
        }
      }
      if (flag) {
        if (filled == length) {
          status = FULL;
          score = num * length;
        } else if (filled == 0) {
          status = EMPTY;
          score = 10 * length;
        } else {
          status = PARTIAL;
          if (num == 0) {
            num = 10;
          }
          score = num * length;
        }
      }

      scale = vars[length - filled];

      ret += scale * score;

      if (status != FULL) ret -= scale * (1 - pow(0.993, blockCount)) * score;

      if (blockCount < 10) {
        if (status == PARTIAL) {
          if (num == lastNum && num != 0 && num != 10) {
            ret -= lastScore * vars[7];
            ret -= sqrt(lastNum / 2);
          }
          lastNum = num;
          lastScore = score;
        } else {
          lastNum = 0;
          lastScore = 0;
        }
        if (status == BROKEN) {
          ret -= sqrt(num);
        }
      }
      if (num != 0 && num != 10 && status == PARTIAL) {
        desired[num] += length - filled;
        waiting[num] += scale * score;
      }
      if (status == PARTIAL) {
        for (int k = 0; k < length; k++) {
          decide[LINES[i][j][k]][0] += 1;
          decide[LINES[i][j][k]][1] += scale * score;
        }
      }
      if (num != 0 && num != 10) {
        needs[num]++;
      }
    }
  }

  printf("after second step, ret = %lf\n", ret);

  for (int i = 1; i <= 9; i++) {
    if (!(desired[i] < 5 || needs[i] < 3)) {
      ret -= pow(desired[i] * vars[6] / 10, 2) * waiting[i];
    }
  }

  printf("after third step, ret = %lf\n", ret);

  scale = pow(blockCount / 20, 2);
  times = 0.4;
  for (int i = 0; i < 20; i++) {
    if (cardList[i][0] == 10) {
      times = 1;
      break;
    }
  }

  printf("after fourth step, ret = %lf\n", ret);

  scale *= times;
  for (int i = 0; i < 20; i++) {
    if (abs(decide[i][0] - 2 < 1e-3)) {
      ret -= scale * vars[8] * decide[i][1];
    } else if (abs(decide[i][0] - 3 < 1e-3)) {
      ret -= scale * vars[9] * decide[i][1];
    }
  }

  printf("after final step, ret = %lf\n", ret);

  return ret;
}

struct StraightStrategy {
  StraightStrategy() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }
    if (game.is_chance()) {
      puts("evaluating game of chance state.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    for (auto action : actions) {
      game.step(action);
      double score = getExpScore(game);
      // printf("trying to put on %d... expScore = %lf\n", (int)action, score);
      if (score > maxScore) {
        maxScore = score;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }

  double evaluate(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }
    if (game.is_chance()) {
      puts("evaluating game of chance state.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    for (auto action : actions) {
      game.step(action);
      double score = getExpScore(game);
      printf("trying to put on %d... expScore = %lf\n", (int)action, score);
      if (score > maxScore) {
        maxScore = score;
      }
      game.redo(action);
    }

    return maxScore;
  }
};

struct DeepStrategy {
  DeepStrategy() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }
    if (game.is_chance()) {
      puts("evaluating game of chance state.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    StraightStrategy straight;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = straight.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }

  double evaluate(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }
    if (game.is_chance()) {
      puts("evaluating game of chance state.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    StraightStrategy straight;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = straight.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return maxScore;
  }
};

struct Deep2Strategy {
  Deep2Strategy() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }
    if (game.is_chance()) {
      puts("evaluating game of chance state.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    DeepStrategy deep;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = deep.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }
};