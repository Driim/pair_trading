# Исследование Mean-Reversing Pair Trading

# Настройка окружения
- Установить [pyenv](https://github.com/pyenv/pyenv)
- Используя pyenv установить miniconda3-latest
- Активировать установленную версию Python `miniconda3-3.11-25.1.1-0`
- Установить необходимые пакеты

## Установка пакетов
```bash
conda install -c ml4t -c jiayi_anaconda -y pandas jupyter matplotlib lxml requests scipy statsmodels numba yfinance xlrd backtrader pyfolio-reloaded
```
# Описание файлов
 - store_tickets - утилиты для получения данных с Yahoo Finance и сохранения в файлы
 - search_pairs - сканирует S&P 500 за 2 года (2023-2024) и ищет коррелирующие и коинтегрированные пары с наилучшим Sharpe ration
 - pair_trading_backtest - подробный backtest пары, search делает это для найденых пар, а тут я просто объяснил этапы и нарисовал картинки
 - trading_pairs - генератор сигналов для текущей стратегии (7 найденых пар), использует полученные в исследовании данные и текущие цены (цену закрытия предыдущего дня) чтобы подготовить сигналы
