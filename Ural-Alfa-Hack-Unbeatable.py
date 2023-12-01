import pandas as pd
import numpy as np

DATES = [
    "2023-09-01",
    "2023-09-02",
    "2023-09-03",
    "2023-09-04",
    "2023-09-05",
    "2023-09-06",
    "2023-09-07",
]

# Определяем константы
MAX_DAY = 7
MIN_VALUE = 500001
MAX_VALUE = 20000000

# загружаем данные по времени работы АТМ, стоимости инкассации, ставке фондирования и ожидаемых изменений остатка
df_atm_info = pd.read_csv("atm_info.csv", sep=";")
df_funding_info = pd.read_csv("funding_rate.csv", sep=";")
df_atm_info["worktime_split"] = df_atm_info["worktime_split"].apply(eval)
df_in_out_train = pd.read_csv("test_private.csv")
df_res = df_in_out_train.copy()
df_res = df_res[["atm_id", "remains"] + DATES]

FOND = (
    np.array(
        df_funding_info[df_funding_info["value_day"].isin(DATES)][
            "funding_rate"
        ].tolist()
    )
    / 365
)


def cost_and_check(prediction, remain, collection_cost, balance_change, days, amt_id):
    """
    Функция для проверки решения на ограничения и подсчета расходов
    """
    total_cost = 0
    balance = remain
    for n in range(len(prediction)):
        if prediction[n] != 0:
            if prediction[n] > MAX_VALUE:
                raise ValueError(f"{prediction[n]}\n{amt_id}\nСлишком много денег")
            if days[n] != 1:
                raise ValueError(f"{amt_id}\nИнкассируем в нерабочий день")
            total_cost += collection_cost
            balance = prediction[n] + balance_change[n]
        else:
            balance += balance_change[n]
        if balance < MIN_VALUE:
            raise ValueError(
                f"{amt_id}\n{balance}\nБаланс меньше 500001\n{balance_change}\n{prediction}"
            )
        total_cost += balance * FOND[n]

    return total_cost


def calculate_cost_without_collection(remain, initial_day, balance_change):
    """
    Функция для подсчета расходов в случае, когда мы не инкассируем, а ждем, когда остаток упадет до минимума
    """
    cost = 0
    costs = []
    for day in range(initial_day, MAX_DAY):
        if remain + balance_change[day] < MIN_VALUE:
            break
        remain += balance_change[day]
        cost += FOND[day] * remain
        costs.append(cost)
    return costs


def calculate_cost_with_collection(
    remain, initial_day, cur_max_day, prediction, work_days, balance_change, inc_cost
):
    """
    Функция для подсчета расходов в случае, когда мы инкассируем каждый день на минимальную сумму для перехода на следующий день
    """
    cost = 0
    costs = []
    for day in range(initial_day, cur_max_day):
        if work_days[day] == 0:
            prediction[day] = 0
            remain += balance_change[day]
        elif day <= 4 and work_days[day + 1] == 0 and work_days[day + 2] == 0:
            cost, remain = handle_two_days_off(
                balance_change, cost, day, inc_cost, prediction, remain
            )
        elif day <= 5 and work_days[day + 1] == 0:
            cost, remain = handle_one_day_off(
                balance_change, cost, day, inc_cost, prediction, remain
            )
        else:
            cost, remain = handle_base_case(
                balance_change, cost, day, inc_cost, prediction, remain
            )
        cost += FOND[day] * remain
        costs.append([cost, prediction, remain])
    return costs


def handle_base_case(balance_change, cost, day, inc_cost, prediction, remain):
    """
    Дополнительная функция для calculate_cost_with_collection для обработки случая без ограничений
    """
    if (
        balance_change[day] + remain >= MIN_VALUE
        and FOND[day] * (balance_change[day] + remain) < inc_cost
    ):
        prediction[day] = 0
        remain += balance_change[day]
    elif balance_change[day] > 0 and remain == MIN_VALUE:
        prediction[day] = 0
        remain += balance_change[day]
    else:
        prediction[day] = (
            abs(balance_change[day]) + MIN_VALUE
            if balance_change[day] < 0
            else 1
            if balance_change[day] >= MIN_VALUE
            else MIN_VALUE
        )
        remain = prediction[day] + balance_change[day]
        if prediction[day] != 0:
            cost += inc_cost
    return cost, remain


def handle_two_days_off(balance_change, cost, day, inc_cost, prediction, remain):
    """
    Дополнительная функция для calculate_cost_with_collection для обработки случая с двумя нерабочими днями подряд
    """
    prediction[day] = 0
    is_minvalue_added = False
    if all(balance_change[d] >= 0 for d in range(day, day + 3)) and remain == MIN_VALUE:
        remain += balance_change[day]
    else:
        for offset in range(3):
            if balance_change[day + offset] < 0:
                prediction[day] += abs(balance_change[day + offset]) + MIN_VALUE
                if is_minvalue_added:
                    prediction[day] -= MIN_VALUE
                is_minvalue_added = True

        if prediction[day] == 0 and remain != MIN_VALUE:
            prediction[day] = 1
        remain = prediction[day] + balance_change[day]
        if prediction[day] != 0:
            cost += inc_cost
    return cost, remain


def handle_one_day_off(balance_change, cost, day, inc_cost, prediction, remain):
    """
    Дополнительная функция для calculate_cost_with_collection для обработки случая с одним нерабочим днем
    """
    if (
        balance_change[day + 1] + balance_change[day] + remain >= MIN_VALUE
        and balance_change[day] + remain >= MIN_VALUE
        and remain == MIN_VALUE
    ):
        prediction[day] = 0
        remain += balance_change[day]
    elif balance_change[day] >= 0:
        if balance_change[day + 1] >= 0 and remain == MIN_VALUE:
            prediction[day] = 0
        else:
            prediction[day] = abs(balance_change[day + 1]) + MIN_VALUE
    else:
        prediction[day] = 0
        is_minvalue_added = False
        if balance_change[day] < 0:
            prediction[day] += abs(balance_change[day]) + MIN_VALUE
            is_minvalue_added = True
        if balance_change[day + 1] < 0:
            prediction[day] += abs(balance_change[day + 1]) + MIN_VALUE
            if is_minvalue_added:
                prediction[day] -= MIN_VALUE

        if prediction[day] == 0 and remain != MIN_VALUE:
            prediction[day] = 1
    remain = prediction[day] + balance_change[day]
    if prediction[day] != 0:
        cost += inc_cost
    return cost, remain


def choose_best_cost(
    inc, remain, prediction, day, days, balance_change, collection_cost
):
    """
    Функция, которая выбирает лучшее решение для данного дня
    """
    remain_temp = inc
    predictions_temp = prediction[::]
    cost_1 = calculate_cost_without_collection(remain_temp, day, balance_change)
    remain_temp = remain
    temp_day = len(cost_1) + day
    cost_2 = calculate_cost_with_collection(
        remain_temp,
        day,
        temp_day,
        predictions_temp,
        days,
        balance_change,
        collection_cost,
    )
    additional_info = 0
    is_second_variant = False
    temp_day = 0
    benefit = 0
    for m in range(len(cost_1)):
        if cost_1[m] > cost_2[m][0]:
            if cost_1[m] - cost_2[m][0] > benefit:
                benefit = cost_1[m] - cost_2[m][0]
                is_second_variant = True
                additional_info = cost_2[m]
                temp_day = day + m
        elif cost_1[m] < cost_2[m][0]:
            if cost_2[m][0] - cost_1[m] > benefit:
                benefit = cost_2[m][0] - cost_1[m]
                is_second_variant = False
                temp_day = day + m
    if is_second_variant:
        return True, additional_info[1][::], temp_day, additional_info[2]
    return [False]


def get_max_collection_sum(balance_change, day):
    """
    Функция рассчёта максимально оптимальной суммы инкассации
    """
    max_consumption = 0
    last_consumption = 0
    consumption_sum_lower_than_max = 0
    consumption_sum = 0
    for k in range(day, MAX_DAY):
        consumption_sum += balance_change[k]
        if consumption_sum < -MAX_VALUE and last_consumption == 0:
            last_consumption = balance_change[k]
            consumption_sum_lower_than_max = abs(consumption_sum)
        max_consumption = min(consumption_sum, max_consumption)
    if last_consumption != 0:
        inc = min(
            min(
                consumption_sum_lower_than_max + last_consumption + MIN_VALUE,
                MAX_VALUE,
            ),
            abs(max_consumption) + MIN_VALUE,
        )
    else:
        inc = min(MAX_VALUE, abs(max_consumption) + MIN_VALUE)
    return inc


def get_prediction_and_cost(table):
    """
    Функция решения задачи, которая также возвращает расходы нашего решения и худшего решения
    """
    cost_cur = 0
    worst_case = 0
    final_prediction = []
    for amt in table.values:
        remain = amt[1]
        days = df_atm_info[df_atm_info["atm_id"] == amt[0]]["worktime_split"].values[0]
        collection_cost = df_atm_info[df_atm_info["atm_id"] == amt[0]][
            "incasationcost"
        ].values[0]
        days = days[4:] + days[:4]
        balance_change = list(map(float, amt[2:]))
        prediction = balance_change[::]
        prev_day = 0
        day = 0
        while True:
            if day >= MAX_DAY:
                break
            if remain + balance_change[day] < MIN_VALUE:
                inc = get_max_collection_sum(balance_change, day)
                best_cost = choose_best_cost(
                    inc, remain, prediction, day, days, balance_change, collection_cost
                )
                if best_cost[0]:
                    prediction, day, remain = best_cost[1:]
                else:
                    remain = inc + balance_change[day]
                    if days[day]:
                        prediction[day] = inc
                    else:
                        prediction[prev_day] = inc
                        day = prev_day
            else:
                best_cost = choose_best_cost(
                    remain,
                    remain,
                    prediction,
                    day,
                    days,
                    balance_change,
                    collection_cost,
                )
                if best_cost[0]:
                    prediction, day, remain = best_cost[1:]
                else:
                    remain += balance_change[day]
                    prediction[day] = 0
                    if days[day] == 1:
                        prev_day = day
            day += 1
        final_prediction.append(prediction)
        cost_cur += cost_and_check(
            prediction,
            amt[1],
            collection_cost,
            balance_change,
            days,
            amt[0],
        )
        worst_case += cost_and_check(
            np.array(days) * MAX_VALUE,
            0,
            collection_cost,
            balance_change,
            days,
            amt[0],
        )
    return final_prediction, cost_cur, worst_case


df_res[DATES], prediction_cost, worst_cost = get_prediction_and_cost(df_res)
df_res = df_res.drop("remains", axis=1)

print(f"Расходы оптимального решения - {prediction_cost}")
print(f"Расходы худшего решения - {worst_cost}")
print(f"Выгода - {worst_cost - prediction_cost}")
df_res.to_csv("data.csv", index=None)
