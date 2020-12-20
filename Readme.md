# Makefile 용어 정리

- $@

현재 Target 이름

- $^

현재 Target이 의존하는 대상들의 전체 목록

- $?

현재 Target이 의존하는 대상들 중 변경된 것들의 목록

- $<

첫 번째 필수 구성 요소의 이름. 대상이 암시적 규칙에서 Recipe를 가져 오는 경우 암시적 규칙에 의해 첫 번째 전체 조건이 됨.

- .PHONY

포니 타겟은 실제 파일 이름을 나타내는 타겟 이름이 아니다. 이것은 명백하게 make 요청을 하는 경우에 실행되는 명령을 위한 목적으로 사용된다. 포니 타겟을 사용하는 이유는 두 가지가 있다고 한다. 첫 번째는 동일한 이름의 파일을 사용한 충돌을 피하기 위함이고, 두 번째는 make 성능향상을 위함이다.

```
clean:
    rm *

.PHONY: clean
```

위와 같이 작성하면, make clean 명령은 clean 이라는 파일이 존재하는지 여부와 상관없이 명령을 수행하게 된다.

- $(lastword names...)

names 인자는 공백으로 나눌 수 있는 이름의 리스트라고 볼 수 있다. 즉, 일련의 name의 나열이다.
lastword는 names에서 마지막 단어를 리턴한다.
즉, 아래의 예시에서 'bar'를 리턴한다.

```
$(lastword foo bar)
```

- $(word n, text)

text의 n 번째 단어를 리턴한다.
즉, 아래의 예시에서 'bar'를 리턴한다.

```
$(word 2, foo bar baz)
```

- .DEFAULT_GOAL

명령 줄에 대상이 지정되지 않은 경우 사용할 기본 목표를 설정합니다. 그리고 하나의 이름만 지정할 수 있으며, 둘 이상의 이름을 지정하면 에러가 발생합니다.

```
# Query the default goal.
ifeq ($(.DEFAULT_GOAL),)
  $(warning no default goal is set)
endif

.PHONY: foo
foo: ; @echo $@

$(warning default goal is $(.DEFAULT_GOAL))

# Reset the default goal.
.DEFAULT_GOAL :=

.PHONY: bar
bar: ; @echo $@

$(warning default goal is $(.DEFAULT_GOAL))

# Set our own.
.DEFAULT_GOAL := foo
```

```
# output
no default goal is set
default goal is foo
default goal is bar
foo
```

# 리눅스 명령어 정리

## head

텍스트로된 파일의 앞부분을 지정한 만큼 출력하는 명령어이다.

```
$ head [-n lines | -c bytes] [file ...]
```

## tail

파일의 마지막 행을 기준으로 지정한 행까지의 파일내용 일부를 출력한다.
리눅스에서 오류나 파일 로그를 실시간으로 확인할 때 매우 유용하게 사용된다.

```
$ tail [option] [filename]
```

- -c, --bytes=K : 줄 단위가 아니라 bytes 단위로 파일의 마지막 부분을 출력한다. -c +K 와 같이 입력하면 파일의 시작부터 K번째 bytes까지 출력한다.

- -f, --follow[={name|descriptor}] : 파일의 마지막부터 10줄을 출력해주고 종료되지 않은채 표준입력을 읽어 들인다.

- -F : 파일 변동시 실시간으로 출력하되 로그파일처럼 특정 시간이 지난 후 파일이 변하게 되면 새로운 파일을 오픈하여 출력한다.

- -n, --lines=K : K 값을 입력하는 경우 마지막 10줄 대신 마지막에서 K 번째 줄까지 출력한다.

- -q : 파일의 이름을 header에 출력하지 않는다.

- -s : -f 옵션과 함께 사용하며, N초(default 1.0)의 시간 간격마다 파일에 새로운 줄이 추가되었는지 검사한다.

- -v : 항상 파일의 이름을 header에 출력한다.

## cut

리눅스에서 파일 내용을 각 필드로 구분하고 필드별로 내용을 추출하며 각 필드들을 구분자로 구분할 수 있는 명령어이다. awk의 print $N 명령어 셋과 유사하나 제한 사항을 가지고 있으며, 스크립트를 작성할 경우 awk보다 더 간편하게 사용이 가능하다. 중요한 옵션으로는 -d(구분자)와 -f(필드 지시자)가 있다. 파일의 각 라인에서 특정 부분을 제거하거나 추출한다.

```
$ cut [option] [filename]
```

- -b, --bytes=LIST : 바이트 단위로 선택

- -c, --characters=LIST : 문자 단위로 선택

- -d, --delimiter=DELIM : 필드를 구분짓는 기본 값은 TAB 대신에 DELIM을 사용

- -f, --fields=LIST : 지정한 필드만을 출력

- -s, --ohnly-delimited : 필드 구분자를 포함하지 않는 줄은 미출력

--output-delimiter=STRING : 출력할때 구분자 대신에 STRING을 사용하며, STRING은 문자나 빈칸등을 사용


