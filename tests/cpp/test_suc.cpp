#include <cstdio>

class A {
public:
    virtual void gao() {
        A::print();
    }

    virtual void print() {
        puts("A");
    }
};

class B: public A {
public:
    void gao() override {
        A::gao();
    }
};

class C: public A {
public:
    void gao() override {
        A::gao();
    }

    void print() override {
        puts("C");
    }
};

int main() {
    A* a = new A();
    A* b = new B();
    A* c = new C();
    a->gao();
    b->gao();
    c->gao();
}