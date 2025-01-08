class BankingSystem:
    def __init__(self):
        self.accounts = {}
        self.pay_id = 0
        self.payments = {}
        self.MILLISECONDS_IN_1_DAY = 24 * 60 * 60 * 1000


    def create_account(self, teimstamp: int, account_id: str) -> bool:
        if account_id in self.accounts:
            return False
        self.accounts[account_id] = {'balance': 0, 'out': 0}
        return True


    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        if account_id not in self.accounts:
            return None
        self.accounts[account_id]['balance'] += amount
        return self.accounts[account_id]['balance']


    def transfer(self, timestamp: int, source_account_id: str, arget_account_id: str, amount: int) -> int | None:
        if source_account_id not in self.accounts or arget_account_id not in self.accounts or source_account_id == arget_account_id or self.accounts[source_account_id]['balance'] < amount:
            return None
        self.accounts[source_account_id]['balance'] -= amount
        self.accounts[source_account_id]['out'] += amount
        self.accounts[arget_account_id]['balance'] += amount
        return self.accounts[source_account_id]['balance']


    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        # print(self.accounts.items())
        return [key for key, value in sorted(self.accounts.items(), key=lambda x: x[1]['out'], reverse=True)]


    def pay(self, timestamp: int, account_id: str, amount: int) -> str | None:
        if account_id not in self.accounts or self.accounts[account_id]['balance'] < amount:
            return None
        self.accounts[account_id]['balance'] -= amount
        self.accounts[account_id]['out'] += amount
        self.pay_id += 1
        id = str('payment' + self.pay_id)
        self.payments[id] = {'account_id': account_id, 'start_time': timestamp, 'amount': amount}
        return id


    def get_payment_status(self, timestamp: int, account_id: str, payment: str) -> str | None:
        if account_id not in self.accounts or payment not in self.payments or self.payments[payment]['account_id'] != account_id:
            return None
        if timestamp < self.MILLISECONDS_IN_1_DAY + self.payments[payment]['start_time']:
            return "IN_PROGRESS"
        else:
            return  "CASHBACK_RECEIVED"
