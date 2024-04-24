# Qubx

## Next generation of Qube quantitative backtesting framework (QUBX)
```          
⠀⠀⡰⡖⠒⠒⢒⢦⠀⠀
⠀⢠⠃⠈⢆⣀⣎⣀⣱⡀  QUBX | Quantitative Backtesting Environment
⠀⢳⠒⠒⡞⠚⡄⠀⡰⠁         (c) 2024, by Dmytro Mariienko
⠀⠀⠱⣜⣀⣀⣈⣦⠃⠀⠀⠀

```                                          
### How to run live trading (Only Binance spot tested)
1. cd experiments/
2. Edit strategy config file (zero_test.yaml). Testing strategy is just doing flip / flop trading once per minute (trading_allowed should be set for trading)
3. Modify accounts config file under ./configs/.env and provide your API binance credentials (see example in example-accounts.cfg):
```
[binance-mde]
apiKey = ...
secret = ...
base_currency = USDT
```
4. Run in console (-j key if want to run under jupyter console)

```
> python -P ..\src\qubx\utils\runner.py configs\zero_test.yaml -a configs\.env -j 
```
