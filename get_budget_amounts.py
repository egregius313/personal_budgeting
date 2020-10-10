#!/usr/bin/env python
"""
Script to dynamically track budgeting goals.

Consumes my GnuCash file and my
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
import numbers
from types import SimpleNamespace
from typing import List, Set, Optional, Tuple, Dict
from typing import io

import piecash
from tabulate import tabulate
import toml

BOOK = piecash.open_book("accounts.gnucash", readonly=True)

STUDENT_LOANS = BOOK.accounts.get(fullname="Liabilities:Student Loans")
FEDERAL_LOANS = BOOK.accounts.get(
    fullname="Liabilities:Student Loans:Federal Student Loans"
)
PARENT_STUDENT_LOANS = BOOK.accounts.get(fullname="Liabilities:Student Loans:Parents")

MASTER_BUDGET = (
    BOOK.session.query(piecash.Budget).filter_by(name="Master Budget").first()
)


class GoalType:
    Aggregate = "aggregate"
    Payoff = "payoff"
    RecurringPercentage = "recurring-percentage"
    Savings = "savings"


class Goal:
    """
    A monetary goal
    """

    predecessors: Set["Goal"]

    completed = False

    def __hash__(self):
        return id(self)

    @property
    def _transitive_predecessors(self):
        to_check = set(self.predecessors)
        seen = set()

        while to_check:
            p = to_check.pop()
            if p in seen:
                continue
            seen.add(p)
            to_check.update(p.predecessors)

        return seen

    @property
    def active(self) -> bool:
        """
        Whether or not the goal should still be considered
        """
        if self.completed:
            return False
        return all(
            not predecessor.active for predecessor in self._transitive_predecessors
        )


@dataclass
class Payoff(Goal):
    """
    A goal which requires paying off
    """

    account: str
    rate: numbers.Real
    periods: int
    predecessors: Set[Goal]

    def monthly_payment(self, income: numbers.Real) -> numbers.Real:
        # print(
        #     f"Balance: {self.remaining} at {self.rate*100}% over {self.periods} periods"
        # )
        return (
            float(self.rate)
            * float(self.remaining)
            / (1 - (1 + self.rate) ** (-self.periods))
        )

    @property
    def completed(self):
        return self.remaining <= 0

    @property
    def remaining(self) -> Decimal:
        """
        The remaining balance necessary for
        """
        try:
            account = BOOK.accounts.get(fullname=self.account)
            return account.get_balance()
        except:
            return Decimal("0.00")

    def __hash__(self):
        return id(self)


@dataclass
class Savings(Goal):
    """
    A goal for building up a savings plan
    """

    account: str
    goal: Decimal
    percentage: Optional[numbers.Real] = None
    max_percentage: Optional[numbers.Real] = None
    predecessors: List[Goal] = None

    def monthly_payment(self, income: numbers.Real) -> numbers.Real:
        if self.percentage:
            return self.percentage * income / 100
        return min(Decimal(0), self.max_percentage * income / 100)

    @property
    def completed(self) -> bool:
        return self.total >= self.goal

    @property
    def total(self) -> Decimal:
        try:
            account = BOOK.accounts.get(fullname=self.account)
            return account.get_balance()
        except:
            return Decimal("0.00")

    @property
    def progress(self):
        return (self.total / self.goal) * 100

    def __hash__(self):
        return id(self)


@dataclass
class Recurring(Goal):
    account: str
    amount: Optional[numbers.Real] = None
    percentage: Optional[numbers.Real] = None
    predecessors: Optional[List[Goal]] = None

    @property
    def completed(self):
        return False

    def monthly_payment(self, income):
        if self.amount:
            return self.amount
        if self.percentage:
            return self.percentage * income / 100
        return Decimal(0)

    def __hash__(self):
        return id(self)


@dataclass
class Aggregate(Goal):
    predecessors: List[Goal]

    def monthly_payment(self, income):
        return 0

    def completed(self):
        return all(pred.completed for pred in self.predecessors)

    def __hash__(self):
        return id(self)


_GOALS = {}


def get_predecessors(account):
    if isinstance(account.get("after"), str):
        return [_GOALS[account["after"]]]
    return [_GOALS[pred] for pred in account.get("after", [])]


def parse_goal(account: dict) -> Dict[str, List[Goal]]:
    type = account["type"]
    name = account["name"]

    if type == GoalType.Aggregate:
        goals = {}
        deps = []
        for dep_goals in map(parse_goal, account["goals"]):
            goals.update(dep_goals)
            deps.extend(dep_goals.values())
        goal = Aggregate(deps)
        goals[name] = goal
        return goals
    if type == GoalType.Payoff:
        return {
            name: Payoff(
                account["account"],
                account["rate"],
                account["periods"],
                get_predecessors(account),
            )
        }
    if type == GoalType.RecurringPercentage:
        return {
            name: Recurring(
                account["account"],
                account.get("amount"),
                account.get("percentage"),
                get_predecessors(account),
            )
        }
    if type == GoalType.Savings:
        return {
            name: Savings(
                account["account"],
                account["goal"],
                account.get("percentage", 0),
                account.get("max_percentage", 100),
                get_predecessors(account),
            )
        }
    return {}


def load_goals(fp: io.IO) -> List[Goal]:
    data = toml.load(fp)

    goals = {}
    for name, goal_info in data.items():
        goal = parse_goal(goal_info)
        goals.update(goal)
        _GOALS[name] = goals.get(goal_info["name"])

    return goals


def _any_budgeted(budget: piecash.Budget, account: piecash.Account) -> bool:
    period = get_monthly_period(budget)
    amounts = BOOK.session.query(piecash.BudgetAmount).filter_by(
        budget=budget, period_num=period
    )
    if amounts.filter_by(account=account).count() > 0:
        return True
    return any(child for child in account.children if amounts.filter_by(account=child))


@dataclass
class Budgeted(Goal):
    account: str
    budget: piecash.Budget

    completed = False

    def monthly_payment(self, income):
        try:
            account = BOOK.accounts.get(fullname=self.account)
            period = get_monthly_period(self.budget)
            amounts = BOOK.session.query(piecash.BudgetAmount).filter_by(
                account=account, budget=self.budget, period_num=period
            )
            return amounts.first().amount
        except:
            return 0

    @property
    def active(self) -> bool:
        try:
            account = BOOK.accounts.get(fullname=self.account)
            return _any_budgeted(self.budget, account)
        except:
            return False

    def __hash__(self):
        return id(self)


def get_monthly_period(budget: piecash.Budget) -> int:
    today = datetime.now()
    start = budget.recurrence.recurrence_period_start
    return (today.year - start.year) * 12 + (today.month - start.month)


class RetirementAccount:
    """
    Retirement account
    """

    account: str
    maximum_contribution: int

    def total_contributions(self, year: Optional[int] = None) -> numbers.Real:
        if not year:
            year = datetime.now().year

        try:
            account = BOOK.accounts.get(fullname=self.account)
        except piecash.GnucashException:
            return 0

        splits = BOOK.session.query(piecash.Split).filter_by(account=account)

        return sum(
            split.value
            for split in splits
            if split.is_debit and split.transaction.post_date.year == year
        )

    def maxed_out(self, year: Optional[int] = None) -> bool:
        """
        """
        return self.total_contributions(year) >= self.maximum_contribution


@dataclass
class _401K(RetirementAccount):
    maximum: int


class RothIRA(RetirementAccount):
    pass


def has_federal_loans() -> bool:
    """
    Whether or not there are outstanding federal lonas
    """
    return FEDERAL_LOANS.get_balance() >= 0


def has_student_loans() -> bool:
    """
    Return whether or not there are any outstanding loans
    """
    return STUDENT_LOANS.get_balance() >= 0


def has_parent_student_loans():
    """
    Return whether or not there are any outstanding parent loans
    """
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a budget")
    parser.add_argument("income", type=float)
    args = parser.parse_args()

    income = args.income

    with open("goals.toml", "r") as f:
        goals = load_goals(f)

    for item in BOOK.session.query(piecash.Account).filter(
        (piecash.Account.type == "EXPENSE") | (piecash.Account.type == "ASSET")
    ):
        goal = Budgeted(item.fullname, MASTER_BUDGET)
        if goal.monthly_payment(income) > 0:
            goals[item.name] = goal

    headers = ["Item", "$", "Percentage"]
    table = []
    total = 0
    for name, goal in goals.items():
        # print(f"Goal {name} is active? {goal.active}")
        if goal.active:
            monthly_payment = float(goal.monthly_payment(income))
            total += monthly_payment
            percentage = "{:.02%}".format(monthly_payment / income)
            table.append((name, "{:.02f}".format(monthly_payment), percentage))

    table.append(("Total", "{:.02f}".format(total), "{:.02%}".format(total / income)))
    table.append(
        (
            "Balance Remaining",
            "{:5.02f}".format(income - total),
            "{:.02%}".format((income - total) / income),
        )
    )

    print(tabulate(table, headers=headers))
