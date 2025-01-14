from banking_system import BankingSystem


# print('----------level 1----------')
# system = BankingSystem()
# print(system.create_account(1, "account1"))
# print(system.create_account(2, "account1"))
# print(system.create_account(3, "account2"))
# print(system.deposit(4, "non-existing", 2700))
# print(system.deposit(5, "account1", 2700))
# print(system.transfer(6, "account1", "account2", 2701))
# print(system.transfer(7, "account1", "account2", 200))

print('----------level 2----------')
system = BankingSystem()
print(system.create_account(1, "account3"))
print(system.create_account(2, "account2"))
print(system.create_account(3, "account1"))
print(system.deposit(4, "account1", 2000))
print(system.deposit(5, "account2", 3000))
print(system.deposit(6, "account3", 4000))
print(system.top_spenders(7, 3))
print(system.transfer(8, "account3", "account2", 500))
print(system.transfer(9, "account3", "account1", 1000))
print(system.transfer(10, "account1", "account2", 2500))
print(system.top_spenders(11, 3))

# print('----------level 3----------')
# system = BankingSystem()
# MILLISECONDS_IN_1_DAY = 24 * 60 * 60 * 1000
# print(system.create_account(1, "account1"))
# print(system.create_account(2, "account2"))
# print(system.deposit(3, "account1", 2000))
# print(system.pay(4, "account1", 1000))
# print(system.pay(100, "account1", 1000))
# print(system.get_payment_status(101, "non-existing", "payment1"))
# print(system.get_payment_status(102, "account2", "payment1"))
# print(system.get_payment_status(103, "account1", "payment1"))
# print(system.top_spenders(104, 2))
# print(system.deposit(3 + MILLISECONDS_IN_1_DAY, "account1", 100))
# print(system.get_payment_status(4 + MILLISECONDS_IN_1_DAY, "account1", "payment1"))
# print(system.deposit(5 + MILLISECONDS_IN_1_DAY, "account1", 100))
# print(system.deposit(99 + MILLISECONDS_IN_1_DAY, "account1", 100))
# print(system.deposit(100 + MILLISECONDS_IN_1_DAY, "account1", 100))

# print('----------level 4----------')
# system = BankingSystem()
# print(system.create_account(1, "account1"))
# print(system.create_account(2, "account2"))
# print(system.deposit(3, "account1", 2000))
# print(system.deposit(4, "account2", 2000))
# print(system.pay(5, "account2", 300))
# print(system.transfer(6, "account1", "account2", 500))
# print(system.merge_accounts(7, "account1", "non-existing"))
# print(system.merge_accounts(8, "account1", "account1"))
# print(system.merge_accounts(9, "account1", "account2"))
# print(system.deposit(10, "account1", 100))
# print(system.deposit(11, "account2", 100))
# print(system.get_payment_status(12, "account2", "payment1"))
# print(system.get_payment_status(13, "account1", "payment1"))
# print(system.get_balance(14, "account2", 1))
# print(system.get_balance(15, "account2", 9))
