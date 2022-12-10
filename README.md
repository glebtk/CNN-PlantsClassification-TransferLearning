##  Crimean Plants Classification

![Example][example]<br>

## О проекте
Проект решает задачу классификации изображений на 50-ти классах растений.

В процессе поиска оптимальной модели для **transfer learning** была 
выбрана модель **MobileNetV3_Large** предварительно обученная 
на датасете **ImageNet**. На данный момент она имеет **точность 0.99** на
тестовой выборке из 1500 изображений растений.

**Датасет** для обучения модели **собран автоматически** путём парсинга 
Яндекс Картинок, и включает в себя, в общей сложности, 13.723 
изображения растений. О датасете подробнее [в этом разделе](#датасет).



<details>
    <summary><strong>Структура проекта</strong></summary>
    
Такую структуру имеет проект *после* выполнения скрипта `download_files.py`, 
который загружает чекпоинт и датасет:
```
.
├── checkpoints                                 
│   └── plants_model_checkpoint
│       └── plants_model.pth.tar                # Чекпоинт модели
│
├── data                                        
│   ├── dataset                                 
│   │   ├── test                                # Директория тестовых изображений
│   │   │   ├── Adenophora liliifolia
│   │   │   ├── Adiantum capillus-veneris
│   │   │               ...
│   │   ├── train                               # Директория тренировочных изображений
│   │   │   ├── Adenophora liliifolia
│   │   │   ├── Adiantum capillus-veneris
│   │   │               ...
│   │   ├── test_labels.csv                     # Тестовый CSV-файл
│   │   └── train_labels.csv                    # Тренировочный CSV-файл
│   └── crimean_plants.csv                      # CSV-файл содержащий названия растений и метку класса
│
├── dataset_preparation                         
│   ├── parsing
│   │   ├── data
│   │   │   └── crimean_plants.csv              # CSV-файл содержащий названия растений и метку класса
│   │   ├── geckodriver
│   │   │   └── geckodriver.exe                 # Драйвер для управления FireFox
│   │   ├── parsing.py                          # Скрипт парсинга растений
│   │   ├── parsing_utils.py                    # Утилиты, необходимые при парсинге
│   │   └── yandex_images_parser.py             # Парсер. (Мой проект с парсером - gitlab.com/gleb_tk/yandex_images_parser)
│   ├── cleaning                                 
│   │   ├── cleaning.py                         # Скрипт очистки датасета 
│   │   └── cleaning_utils.py                   # Утилиты, необходимые при очистке датасета
│   └── labeling
│       ├── labeling.py                         # Скрипт разметки данных
│       └── labeling_utils.py                   # Утилиты, необходимые при очистке данных
│
├── config.py                                   # Конфигурации
├── dataset.py                                  # Класс датасета
├── download_files.py                           # Скрипт, загружающий датасет и чекпоинт модели
├── hyperparameter_optimization.py              # Скрипт поиска оптимальных гиперпараметров 
├── model.py                                    # Класс модели
├── train.py                                    # Скрипт обучения модели
├── utils.py                                    # Утилиты 
├── requirements.txt                            # Зависимости проекта
└── README.md
```
</details>


## Технические решения

#### Основные технологии

- **[Python 3.9](https://www.python.org/downloads/release/python-390/)**<br> 
- **[PyTorch](https://pytorch.org)** - фреймворк глубокого обучения.<br>
- **[TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)** - отслеживание процесса обучения в реальном времени.<br> 
- **[Optuna](https://optuna.org)** - поиск оптимальных гиперпараметров.<br>
- **[Selenium](https://www.selenium.dev)** - парсинг датасета с Яндекс Картинок.
- **[Pandas](https://pandas.pydata.org)** - работа с CSV-файлами.<br><br>


#### Transfer learning

Поскольку стояла задача добиться высокой точности на маленьком датасете,
было принято решение 
использовать **предобученную модель**.

<details>
    <summary><strong>Список опробованных моделей</strong></summary>

```python
models = ["alexnet", "convnext_tiny", "densenet121", "densenet201", "resnet18", "mobilenet_v3_small", "mobilenet_v3_large"]
```

</details>

Из выбранных **7-ми** моделей 
наиболее точные результаты показала 
[ConvNeXt_Tiny](https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html),
доступная по умолчанию в torchvision. 
Однако, ввиду её тяжеловесности, предпочтение было отдано модели
[MobileNet_V3_Large](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html),
которая демонстрировала сравнимые по точности результаты, но при этом имела 
**в 5.2 раза меньше параметров**.

Последний полносвязный слой классификатора в MobileNet был заменен: 
```python
# Заменяем последний слой:
in_features = mobilenet.classifier[-1].in_features
mobilenet.classifier[-1] = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)
```
Так как изначально сеть обучалась на датасете ImageNet, который содержит
1000 классов, количество выходных нейронов у нее было равно тысяче.
После замены слоя, их количество стало равняться 50-ти. 
Была добавлена возможность "заморозки" градиентов в feature extractor.<br>
В остальном модель оставлена без изменений.<br><br>


#### Парсинг изображений и подготовка датасета

Для **парсинга** изображений был использован ранее написанный мной [парсер
Яндекс Картинок](https://gitlab.com/gleb_tk/yandex_images_parser).<br>
Применительно к отдельно взятому растению, производился поиск 200
изображений по его латинскому названию. Далее **для каждого** из первых 20 
результатов выдачи производился поиск 40 похожих изображений.  

На этапе **очистки** датасета все изображения проходили проверку
на соответствие условиям, удалялись дубликаты.
После чего изображения были отмасштабированы.

Подробнее про **алгоритм очистки** можно прочитать под спойлером:

<details>
    <summary><strong>Алгоритм очистки датасета</strong></summary>

Для изображений, относящихся к одному классу, применяем 
следующие шаги:

1. Проверяем, все ли изображения являются действительными, удаляем мусор.
2. Удаляем изображения, которые не удовлетворяют заданным условиям 
(размер >= 256 пикселей по меньшей стороне, цветовой режим RGB)
3. Удаляем дубликаты изображений. Для этого, с использованием библиотеки
[ImageHash](https://pypi.org/project/ImageHash/), получаем 
[average_hash](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html) 
избражений и удаляем изображения имеющие 
одинаковые или близкие хэши.
4. Масштабируем изображения до 256px по меньшей стороне.
5. Сохраняем изображения в отдельную директорию.
6. Готово! Переходим к следующему классу.

Финальную очистку производим вручную.

</details>

При **разметке** изображения разбивались на тренировочную и тестовую выборки.
Пути к изображениям и метки класса сохранялись в CSV-файл.

## Датасет

Датасет содержит **13.723 цветных изображения** растений относящихся к 50 классам.
Из них **12.223** составляют тренировочную выборку, и **1.500** тестовую.

<details>
    <summary><strong>Структура датасета</strong></summary>

```
dataset                                 
├── test                                        # Директория тестовых изображений
│   ├── Adenophora liliifolia                   # Директория 1-го класса
│   │   ├──adenophora_liliifolia_test_1.jpg     
│   │   ├──adenophora_liliifolia_test_2.jpg
│   │               ...
│   │               
│   ├── Adiantum capillus-veneris               # Директория 2-го класса
│               ...
│
├── train                                       # Директория тренировочных изображений
│   ├── Adenophora liliifolia                   # Директория 1-го класса
│   │   ├──adenophora_liliifolia_train_1.jpg
│   │   ├──adenophora_liliifolia_train_2.jpg
│   │               ...
│   │
│   ├── Adiantum capillus-veneris               # Директория 2-го класса
│               ...
│
├── test_labels.csv                             # Тестовый CSV-файл содержащий столбцы path и label (путь к изображению и метка класса)
└── train_labels.csv                            # Тренировочный CSV-файл содержащий столбцы path и label
```

</details>

Пример случайных изображений из датасета:

![Dataset](https://i.imgur.com/1ZqRspx.png)

- Каждое изображение относится только к **одному** классу. 
- В среднем, на класс приходится **274** изображения.
- Минимальный размер изображения: **256x256 px**.
- Среднее по датасету (mean):   `[0.4074, 0.4307, 0.2870]`.
- Стандартное отклонение (std): `[0.2128, 0.2006, 0.2053]`.

Датасет в доступен по прямой ссылке: 
[dataset.zip](https://gitlab.com/glebtutik/crimean_plants_classification_files/-/raw/main/data/dataset.zip)
(~274MB).

## Процесс обучения и результаты

В процессе поиска были найдены следующие гиперпараметры:

```python
BATCH_SIZE = 90
LEARNING_RATE = 9e-05
```

Модель обучалась в течении 20-ти эпох.
Лучшая точность на тестовой выборке составила **0.99** (99%), и была 
достигнута после 18-й эпохи.

Ниже предоставлена confusion matrix и графики обучения модели.

<details>
    <summary><strong>Графики и матрица</strong></summary>

**Графики**, полученные в процессе обучения (accuracy, loss):

![Accuracy, Loss](https://i.imgur.com/caJE8S3.jpg)

**Confusion matrix** (матрица несоответствий):

![Confusion matrix](https://i.imgur.com/XGxscVO.jpg)

В большем резмере матрицу можно посмотреть по 
[ссылке](https://i.imgur.com/Nhkkeex.jpg).

</details>


## Как запустить проект?

1. Клонируйте репозиторий:
```bash
$ git clone https://gitlab.com/gleb_tk/crimean_plants_classification.git
```

2. Перед использованием необходимо установить зависимости проекта.<br>
**Перейдите** в директорию склонированного репозитория 

```bash
$ cd путь/к/директории
```

и выполните следующую команду:

```bash
$ pip install -r requirements.txt
```

3. Убедитесь, что все зависимости успешно установлены.
4. Выполните скрипт `download_files.py`. Он загрузит датасет и 
чектоинт модели в правильные места:

```bash
$ python download_files.py
```

5. Готово! Можно запускать проект.

## Способы связи

[![Mail](https://i.imgur.com/HILZFT2.png)](mailto:tutikgv@gmail.com)
**E-mail:**
[tutikgv@gmail.com](mailto:tutikgv@gmail.com) <br>

[![Telegram](https://i.imgur.com/IMICyTA.png)](https://t.me/glebtutik)
**Telegram:**
https://t.me/glebtutik <br>



[cover]: https://i.imgur.com/D7So6VS.png "Cover"
[example]: https://i.imgur.com/jmO3u0U.gif "Example"


