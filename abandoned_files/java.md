---
layout: notes
section-type: notes
title: Java Basics
category: en_misc
---

* TOC
{:toc}
---

## Data Type
<hr>

### Integer
* Difference between int and Integer

```java
Integer n = 9;
Integer n = Integer.valueOf(9);
// ALSO
int a = Intger.parseInt("10");
```

Those codes above are valid

```java
int n = 10; //valid
int a = int.parseInt("10"); //Not valid
```


### Create an Array

```java
int intArray[];
intArray = new int[20];
```

OR

```java
int [] intArray = new int[20];
```

<br>
<br>
<br>

## Generics
<hr>

* From the point of my view, I think **Generics** is just another kind of class.

* For **Integer** class, it implements method compareTo, and so on. Therefore, the container like ArrayList can use the Generics \<String> \<Integer> .etc to create it.

[A video to watch](https://www.youtube.com/watch?v=9tHLV0u87G4)

```java
List<T> values = new ArrayList<T>();
values.add(7);
values.add("Navin");
```

Another Example in Code

```java
Integer[] a = {1,2,3,4,5};
Character[] b = {'b', 'c', 'd'};
printMe(a);
printMe(b);
//Another Method
public static<T> void printMe(T[] x){
    // T here stands for Type correspond with <T>
    for(T b:x){
        System.out.printf("%s ", b);
    }
}
```


### Name convetion about Generics in Java
* E - Element
* K - Key
* N - Number
* V - Value
* T - Type

### Generics Application in Java
* Say we have different types of data and we want to return a sorted list. Therefore, we need a class called "compareTo", and to use generics

* Following codes means that only the objects inherit from comparable class can be used in this method.

* Tips: \<T> means we will use it in method
* Tips: T means we will return Type of T in this method


```java
<T extends Comparable<T>>
// this adds boundary to T
```

```java
<T super Number>
// this adds lower boundary to T
// T can be Number or More Generics
```


```java
public static <T extends Comparable<T>> T max(T[] x){
    int size  = x.length;
    T max = T[size];
    for (int i=0; i<size-1; i++){
        if(x[i].compareTo(x[i+1])>0){
            max = x[i];
        }
    }
    return max;
}
```

<br>
<br>
<br>

## Java Control Flow
<hr>

### For Loop

```java
Integer[] array = {1,2,3,4};
Integer[] 
for(Integer i: array){

}
```

<br>
<br>
<br>

## Java Class
<hr>

### Inheritence
* subclass can **extends** superclass
* subclass can call methods in superclass
* Private means that subclass cannot inheriate methods or certain things.

### Override
* subclass can recreate certain methods in superclass.
* Overide just need to create same method in superclass.

### Interface
* Write an interface named "Animal"
* Can only write abstract methods in interface
* Implements it

```java
public dog implements Animal, comparable<T>{
    // Here is the place you need to implements 
    //those methods in that interface with 
    //abstract methods
}
```
### Abstract classes and methods
* Abstract classes cannot form an instance, thus the code below is invalid
* Abstract classes exist to be extended, they cannot be instantiated.

```java
// GameObject is an abstract class
GameObject game = new GameObject();// this sentence will be invalid in java with syntax error
```

* If we add an abstract method in abstract class, and if we extend the abstract class in another class; then we need to implement those abstract methods. Otherwise, we will have same error as we did not implement methods for interfaces.


### Polymorphism

```java
public abstract class GameObject {
	public abstract void draw(); 
	
	public static void main(String args[]) {
		GameObject player = new Player();
		GameObject menu = new Menu();
		
		GameObject[] gameobjects = new GameObject[2];
		// these instances response to code above
		// the polymorphism allows us to call same methods
		gameobjects[0] = player;
		gameobjects[1] = menu;
		// in every iteration, the obj corresponds to instances above
		for(GameObject obj:gameobjects) {
			obj.draw();
		}
	}
}
```

### Constructor
* **Contructor** has the same name as the class
* Therefore, if we add a constructor like this

```java
public class Tuna{
    private String girlName;
    public Tuna(String name){
        this.girlName = name;
    }
    public void saying() {
    	System.out.println(this.girlName);
    }
}
//main method:
Tuna tuna = new Tuna("Nancy");
tuna.saying();
```

```
output: Nancy.
```

<br>
<br>
<br>

## Collection
<hr>

### Interface in Collection
* Iterator
    * oublic boolean hasNext()
    * public Object next()
    * public void remove()

* Iterable
    * Iterator\<T> iterator()

* Collection
    * Boolean add(Object obj)
    * Boolean addAll(Object obj)
    * void clear()
    * ...


### List
* ArrayList
* LinkedList
    * Singly LinkedList(1 Direction Pointers)
    * Doubly LinkedList(2 Direction Pointers)
* Vector

We can use following codes to add any type of data into the arraylist

```java
List<Object> list1 = new ArrayList();
```

```java
String[] things = {"eggs", "lasers", "hats"};
Integer[] matter = {1,2,3,23,34};
List<Object> list1 = new ArrayList();
for (String x: things) {
    list1.add(x);
}
for (Integer y: matter) {
    list1.add(y);
}
```

* Interesting things is that we can also add an instance to an ArrayList

```java
List<Object> list = new ArrayList<>();
Student s1 = new Student();
s1.name = "Heming";
s1.age = 22;
list.add(s1);
```

### Iterate in ArrayList
* foreach

```java
for(Object item: list){}

```

* iterator

```java
Iterator<Object> iter = list.iterator();
while(iter.hasNext()){
    T item = iter.next();
    System.out.println(item);
}
```

### Queue
* FIFO: First In First Out
* The first element will be removed first
* The last element will be removed in the end

<center>
<img src=".//enmisc_pictures/java0001.png" height="75%" width="75%">
</center>

```java
Queue<Integer> q = new LinkedList<>();
```

* Difference Between PQ(Priority Queue) and Q(Queue)

<center>
<img src=".//enmisc_pictures/java0002.png" height="100%" width="100%">
</center>

### Java Sets
* Sets focus on uniqueness
* We cannot add duplicate element in Sets
* Three Foundemental Sets
    * HashSet
    * LinkedHashSet
    * TreeSet

* HashSet
    * Using hash function
    * HashSet h = new HashSet();

* LinkedHashSet
    * Maintain the insertion order but not random like HashSet

* TreeSet
    * Sorted Version of your set

<center>
<img src=".//enmisc_pictures/java0003.png" height="100%" width="100%">
</center>

* The image above shows that
    * HashSet does not duplicate
    * HashSet does not have index for them, therefore, element is in random order.


### HashMap
* map.put() Input key value pair
* map.keySet() Get the key in sets
* map.get(key) Fetch the value in pairs

```java
Map<String, String> map = new HashMap<>();
map.put("myName", "Navin");
map.put("actor", "John");
map.put("ceo","Marisa");

Set<String> keys = map.keySet();
for(String key: keys) {
    System.out.println(map.get(key));
}
```
