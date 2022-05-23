use std::io::Write;
use itertools::PeekNth;
use std::slice::Iter;
use std::collections::HashMap;

/*

Grammar:

symbol := Symbol s

term := Number n
      | Sub <term>
      | LParen <expr> RParen
      | <symbol>

call := <term> (LParen <expr_list> RParen)?

dot_operator := <call> {Dot <call>}*

pow := <dot_operator> {Pow <dot_operator>}*

factor := <pow> {(Mul|Div) <pow>}*

addsub_expr := <factor> {(Add|Sub) <factor>}*

bool_expr := <addsub_expr> {(Equals|LT|GT|LTE|GTE|NotEquals|) <addsub_expr}*

expr := <bool_expr>

expr_list := <expr> {Comma <expr>}*

statement := <assign_statement>
           | <if_statement>
           | <for_statement>
           | <while_statement>
           | <expr_statement>

expr_statement := <expr>

assign_statement := <symbol> Equiv <expr>

body := (<block> | <statement>)

if_statement := If LParen <expr> RParen <body>

for_statement := For LParen <assign_expr> Semicolon <expr> Semicolon <expr> RParen <body>

while_statment While LParen <expr> Rparen <body>

statement_list := <statement> {Semicolon <statement>}*

block := LCParen <statement_list> RCParen

function_definition := Function <symbol> LParen (<symbol> {Comma <symbol>}*)? RParen <block>

program := <function_definition>+

*/

enum Token {
    Number(i64),
    Text(String),
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
    Pow,
    Dot,
    Equals,
    LessThan,
    GreaterThan,
    LessThanEquals,
    GreaterThanEquals,
    NotEquals,
    Comma,
    If,
    For,
    While,
    Equiv,
    LCParen,
    RCParen,
    Semicolon,
    Function
}

fn is_digit_opt(c: Option<&char>) -> bool {
    match c {
        Some('0'..='9') => true,
        _ => false,
    }
}

fn is_alpha_opt(c: Option<&char>) -> bool {
    match c {
        Some('a'..='z' | 'A'..='Z') => true,
        _ => false,
    }
}

fn tokenise(program: String) -> Vec<Token> {
    let mut tokens = Vec::new();

    let mut c = itertools::peek_nth(program.chars());

    loop {
        if is_digit_opt(c.peek()) {
            let mut acc: i64 = 0;

            while is_digit_opt(c.peek()) {
                acc *= 10;
                acc += (c.next().unwrap() as i64) % 0x30;
            }
            tokens.push(Token::Number(acc))
        } else if is_alpha_opt(c.peek()) {
            let mut acc: String = String::from("");

            while is_alpha_opt(c.peek())
                || is_digit_opt(c.peek())
                || c.peek() == Some(&'_')
                || c.peek() == Some(&'-')
            {
                acc.push(c.next().unwrap());
            }
            match acc.as_str() {
                "fn" => tokens.push(Token::Function),
                "if" => tokens.push(Token::If),
                "for" => tokens.push(Token::For),
                "while" => tokens.push(Token::While),
                
                _ => tokens.push(Token::Text(acc))
            }
        } else {
            if c.peek() == None {
                break;
            }

            match c.peek().unwrap() {
                '+' => tokens.push(Token::Add),
                '-' => tokens.push(Token::Sub),
                '*' => tokens.push(Token::Mul),
                '/' => tokens.push(Token::Div),
                '(' => tokens.push(Token::LParen),
                ')' => tokens.push(Token::RParen),
                '^' => tokens.push(Token::Pow),

                '.' => tokens.push(Token::Dot),
                ',' => tokens.push(Token::Comma),
                '{' => tokens.push(Token::LCParen),
                '}' => tokens.push(Token::RCParen),

                ';' => tokens.push(Token::Semicolon),
                
                '>' => match c.peek_nth(1) {
                    Some('=') => {
                        c.next();
                        tokens.push(Token::GreaterThanEquals);
                    }
                    _ => tokens.push(Token::GreaterThan)
                }

                '<' => match c.peek_nth(1) {
                    Some('=') => {
                        c.next();
                        tokens.push(Token::LessThanEquals);
                    }
                    _ => tokens.push(Token::LessThan)
                }

                '=' => match c.peek_nth(1) {
                    Some('=') => {
                        c.next();
                        tokens.push(Token::Equals);
                    }
                    _ => tokens.push(Token::Equiv)
                }

                ' ' | '\n' | '\t' => (),
                t @ _ => panic!("Unexpected token: `{}`!", t),
            }

            c.next();
        }
    }

    return tokens;
}

fn print_tokens(tokens: &Vec<Token>) {
    tokens.iter().for_each(|t| {
        println!(
            "{}",
            match t {
                Token::Number(_) => "Number",
                Token::Add => "Add",
                Token::Sub => "Sub",
                Token::Mul => "Mul",
                Token::Div => "Div",
                Token::LParen => "LParen",
                Token::RParen => "RParen",
                Token::Pow => "Pow",

                Token::Text(_) => "Text",

                _ => "Unknown Token",
            }
        )
    });
}

#[derive(Clone)]
enum AST {
    Number(i64),
    Symbol(String),
    Add(Box<AST>, Box<AST>),
    Sub(Box<AST>, Box<AST>),
    Mul(Box<AST>, Box<AST>),
    Div(Box<AST>, Box<AST>),
    Pow(Box<AST>, Box<AST>),
    Negate(Box<AST>),

    Call(Box<AST>, Box<AST>),
    Dot(Box<AST>, Box<AST>),

    Program(Box<Vec<Box<AST>>>),
    FunctionDef(Box<AST>, Box<Vec<AST>>, Box<AST>),
    StatementList(Box::<Vec<Box<AST>>>),
    WhileStatement(Box<AST>, Box<AST>),
    ForStatement(Box<AST>, Box<AST>, Box<AST>, Box<AST>),
    IfStatement(Box<AST>, Box<AST>),
    Assign(Box<AST>, Box<AST>),
    ExprList(Box<Vec<Box<AST>>>),
    NotEquals(Box<AST>, Box<AST>),
    GreaterThanEquals(Box<AST>, Box<AST>),
    LessThanEquals(Box<AST>, Box<AST>),
    GreaterThan(Box<AST>, Box<AST>),
    LessThan(Box<AST>, Box<AST>),
    Equals(Box<AST>, Box<AST>),
}

macro_rules! eat(($e:expr, $p:pat) => {
    match $e.peek() {
        Some($p) => $e.next(),
        _ => panic!("Syntax error!")
    }
});

macro_rules! handles { 
    ( $self:expr, $prev:expr, $( ($m:pat, $t:expr) ),* ) => {
        {
            let mut ast = $prev;
        
            loop {
                match $self.tokens.peek() {
                    $(
                        Some($m) => {
                            eat!($self.tokens, $m);
                            ast = $t(Box::new(ast), Box::new($prev));
                        }
                    )*
                    _ => break,
                }
            }
            
            return ast;
        }
    }
}

struct Parser<'b> {
    tokens: &'b mut PeekNth<Iter<'b, Token>>
}

impl<'b> Parser<'b> {

    pub fn parse<'a>(tokens: &'b mut PeekNth<Iter<'b, Token>>) -> AST {
        return (Self {tokens}).program();
    }

    fn symbol(&mut self) -> AST {
        if let Some(Token::Text(symbol_name)) = eat!(self.tokens, Token::Text(_)) {
            return AST::Symbol((symbol_name).clone());
        }
        panic!();
    }

    fn term(&mut self) -> AST {
        let ast;

        match self.tokens.peek() {
            Some(Token::Number(n)) => {
                eat!(self.tokens, Token::Number(_));
                ast = AST::Number(*n);
            }
            Some(Token::Sub) => {
                eat!(self.tokens, Token::Sub);
                ast = AST::Negate(Box::new(self.term()));
            }
            Some(Token::LParen) => {
                eat!(self.tokens, Token::LParen);
                ast = self.expr();
                eat!(self.tokens, Token::RParen);
            }
            _ => ast = self.symbol()
        }

        return ast;
    }

    fn call(&mut self) -> AST {
        let mut ast = self.term();

        match self.tokens.peek() {
            Some(Token::LParen) => {
                eat!(self.tokens, Token::LParen);
                ast = AST::Call(Box::new(ast), Box::new(self.expr_list()));
                eat!(self.tokens, Token::RParen);
            }
            _ => ()
        }
        return ast;
    }

    fn dot_operator(&mut self) -> AST {
        handles!(self, self.call(), (Token::Dot, AST::Dot));
    }

    fn power(&mut self) -> AST {
        handles!(self, self.dot_operator(), (Token::Pow, AST::Pow));
    }

    fn factor(&mut self) -> AST {
        handles!(self, self.power(), 
            (Token::Mul, AST::Mul),
            (Token::Div, AST::Div)
        );
    }

    fn addsub_expr(&mut self) -> AST {
        handles!(self, self.factor(), 
            (Token::Add, AST::Add),
            (Token::Sub, AST::Sub)
        );
    }

    fn bool_expr(&mut self) -> AST {
        handles!(self, self.addsub_expr(), 
            (Token::Equals, AST::Equals),
            (Token::LessThan, AST::LessThan),
            (Token::GreaterThan, AST::GreaterThan),
            (Token::LessThanEquals, AST::LessThanEquals),
            (Token::GreaterThanEquals, AST::GreaterThanEquals),
            (Token::NotEquals, AST::NotEquals)
        );
    }

    fn expr(&mut self) -> AST {
        return self.bool_expr();
    }

    fn expr_list(&mut self) -> AST {
        if let Some(Token::RParen) = self.tokens.peek() {
            return AST::ExprList(Box::new(Vec::new()));
        } else {
            let mut exprs = vec![Box::new(self.expr())];

            loop {
                match self.tokens.peek() {
                    Some(Token::Comma) => {
                        eat!(self.tokens, Token::Comma);
                        exprs.push(Box::new(self.expr()));
                    }
                    _ => break
                }
            }

            return AST::ExprList(Box::new(exprs));
        }
    }

    fn statement(&mut self) -> AST {
        match self.tokens.peek() {
            Some(Token::If) => return self.if_statement(),
            Some(Token::For) => return self.for_statement(),
            Some(Token::While) => return self.while_statement(),
            _ => match self.tokens.peek_nth(1) {
                Some(Token::Equiv) => return self.assign_statement(),
                _ => return self.expr_statement()
            }
        }
    }

    fn expr_statement(&mut self) -> AST {
        return self.expr();
    }

    fn assign_statement(&mut self) -> AST {
        let base = self.symbol();
        eat!(self.tokens, Token::Equiv);
        return AST::Assign(Box::new(base), Box::new(self.expr()));
    }

    fn body(&mut self) -> AST {
        match self.tokens.peek() {
            Some(Token::LCParen) => return self.block(),
            _ => return self.statement()
        }
    }

    fn if_statement(&mut self) -> AST {
        eat!(self.tokens, Token::If);
        eat!(self.tokens, Token::LParen);
        let expr = self.expr();
        eat!(self.tokens, Token::RParen);
        return AST::IfStatement(Box::new(expr), Box::new(self.body()));
    }

    fn for_statement(&mut self) -> AST {
        eat!(self.tokens, Token::For);
        eat!(self.tokens, Token::LParen);
        let init = self.expr();
        eat!(self.tokens, Token::Semicolon);
        let comp = self.expr();
        eat!(self.tokens, Token::Semicolon);
        let run = self.expr();
        eat!(self.tokens, Token::RParen);
        return AST::ForStatement(Box::new(init), Box::new(comp), Box::new(run), Box::new(self.body()));
    }

    fn while_statement(&mut self) -> AST {
        eat!(self.tokens, Token::While);
        eat!(self.tokens, Token::LParen);
        let expr = self.expr();
        eat!(self.tokens, Token::RParen);
        return AST::WhileStatement(Box::new(expr), Box::new(self.body()));
    }

    fn statement_list(&mut self) -> AST {
        let mut statements = vec![Box::new(self.statement())];

        loop {
            match self.tokens.peek() {
                Some(Token::Semicolon) => {
                    eat!(self.tokens, Token::Semicolon);
                    statements.push(Box::new(self.statement()));
                }
                _ => break
            }
        }

        return AST::StatementList(Box::new(statements));
    }

    fn block(&mut self) -> AST {
        eat!(self.tokens, Token::LCParen);
        let ast = self.statement_list();
        eat!(self.tokens, Token::RCParen);
        return ast;
    }

    fn function_definition(&mut self) -> AST {
        eat!(self.tokens, Token::Function);
        let name = self.symbol();
        eat!(self.tokens, Token::LParen);
        let mut params = vec![];
        if matches!(self.tokens.peek(), Some(Token::Text(_))) {
            params.push(self.symbol());
            loop {
                match self.tokens.peek() {
                    Some(Token::Comma) => {
                        eat!(self.tokens, Token::Comma);
                        params.push(self.symbol());
                    }
                    _ => break
                }
            }
        }
        eat!(self.tokens, Token::RParen);
        return AST::FunctionDef(Box::new(name), Box::new(params), Box::new(self.block()));
    }

    fn program(&mut self) -> AST {
        let mut funcs = vec![];
        funcs.push(Box::new(self.function_definition()));
        loop {
            match self.tokens.peek() {
                Some(Token::Function) => funcs.push(Box::new(self.function_definition())),
                _ => break
            }
        }
        return AST::Program(Box::new(funcs));
    }
}

#[derive(Clone)]
enum Value {
    Number(i64),
    Function(AST),
    Bool(bool),
    String(String)
}

struct Interpreter<'a> {
    symbolmap: &'a mut HashMap<String, Value>
}

impl<'a> Interpreter<'a> {
    pub fn run(ast: AST) {
        let mut ht = HashMap::new();
        let mut s = Interpreter {symbolmap: &mut ht};
        s.eval(ast);
        s.exec_func(String::from("main"));
    }

    fn exec_func(&mut self, func_name: String) {
        match self.symbolmap.get(&func_name) {
            Some(Value::Function(ast)) => {
                self.eval(ast.clone());
                ()
            }
            _ => ()
        }
    }

    fn truthy(v : Value) -> bool {
        match v {
            Value::Bool(b) => b,
            Value::Number(0) => false,
            Value::String(s) => match s.as_str() {
                "" => false,
                _ => true
            }
            _ => true
        }
    }

    fn eval(&mut self, ast: AST) -> Option<Value> {
        match ast {
            AST::Symbol(s) => {
                match self.symbolmap.get(&s) {
                    Some(v) => return Some(v.clone()),
                    _ => None
                }
            }

            AST::Number(n) => Some(Value::Number(n)),
            AST::Negate(x) => {
                if let Some(Value::Number(n)) = self.eval(*x) {
                    return Some(Value::Number(-n));
                }
                None
            }
            AST::Add(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                if let (Some(Value::Number(x)), Some(Value::Number(y))) = (lhs, rhs) {
                    return Some(Value::Number(x + y));
                }
                None
            }
            AST::Sub(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                if let (Some(Value::Number(x)), Some(Value::Number(y))) = (lhs, rhs) {
                    return Some(Value::Number(x - y));
                }
                None
            }
            AST::Mul(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                if let (Some(Value::Number(x)), Some(Value::Number(y))) = (lhs, rhs) {
                    return Some(Value::Number(x * y));
                }
                None
            }
            AST::Div(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                if let (Some(Value::Number(x)), Some(Value::Number(y))) = (lhs, rhs) {
                    return Some(Value::Number(x / y));
                }
                None
            }
            AST::Pow(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                if let (Some(Value::Number(x)), Some(Value::Number(y))) = (lhs, rhs) {
                    return Some(Value::Number(x + y));
                }
                None
            }

            AST::Equals(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                return match (lhs, rhs) {
                    (Some(Value::Number(x)), Some(Value::Number(y))) => Some(Value::Bool(x == y)),
                    (Some(Value::Bool(x)), Some(Value::Bool(y))) => Some(Value::Bool(x == y)),
                    (Some(Value::String(x)), Some(Value::String(y)))=> Some(Value::Bool(x == y)),
                    _ => Some(Value::Bool(false))
                }
            }

            AST::NotEquals(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                return match (lhs, rhs) {
                    (Some(Value::Number(x)), Some(Value::Number(y))) => Some(Value::Bool(x != y)),
                    (Some(Value::Bool(x)), Some(Value::Bool(y))) => Some(Value::Bool(x != y)),
                    (Some(Value::String(x)), Some(Value::String(y)))=> Some(Value::Bool(x != y)),
                    _ => Some(Value::Bool(false))
                }
            }

            AST::LessThan(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                return match (lhs, rhs) {
                    (Some(Value::Number(x)), Some(Value::Number(y))) => Some(Value::Bool(x < y)),
                    _ => Some(Value::Bool(false))
                }
            }

            AST::GreaterThan(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                return match (lhs, rhs) {
                    (Some(Value::Number(x)), Some(Value::Number(y))) => Some(Value::Bool(x <= y)),
                    _ => Some(Value::Bool(false))
                }
            }

            AST::LessThanEquals(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                return match (lhs, rhs) {
                    (Some(Value::Number(x)), Some(Value::Number(y))) => Some(Value::Bool(x > y)),
                    _ => Some(Value::Bool(false))
                }
            }

            AST::GreaterThanEquals(x, y) => {
                let lhs = self.eval(*x);
                let rhs = self.eval(*y);

                return match (lhs, rhs) {
                    (Some(Value::Number(x)), Some(Value::Number(y))) => Some(Value::Bool(x >= y)),
                    _ => Some(Value::Bool(false))
                }
            }

            AST::FunctionDef(symbol, params, block) => {
                if let AST::Symbol(name) = *symbol {
                    self.symbolmap.insert(name, Value::Function(*block));
                }
                None
            }
            
            AST::Program(function_definitions) => {
                for fd in *function_definitions {
                    self.eval(*fd);
                }
                None
            }

            AST::StatementList(statements) => {
                for s in *statements {
                    self.eval((*s).clone());
                }
                None
            }

            AST::Call(name, exprlist) => {
                let mut rval = None;
                if let (AST::Symbol(s), AST::ExprList(el)) = (*name, *exprlist) {
                    match s.as_str() {
                        "print" => {
                            for e in *el {
                                match self.eval((*e).clone()) {
                                    Some(Value::Number(n)) => {
                                        print!("{} ", n);
                                    }
                                    Some(Value::String(s)) => {
                                        print!("{} ", s);
                                    }
                                    Some(Value::Bool(b)) => {
                                        print!("{} ", if b {"true"} else {"false"});
                                    }
                                    _ => ()
                                }
                            }
                            println!();
                        }
                        func_name => if let Some(Value::Function(ast)) = self.symbolmap.get(&func_name.to_string()) {
                            self.eval(ast.clone());
                        }
                    }
                }
                rval
            }

            AST::Assign(symb, expr) => {
                if let AST::Symbol(s) = *symb {
                    if let Some(r) = self.eval(*expr) {
                        self.symbolmap.insert(s, r);
                    }
                }
                None
            }

            _ => panic!("Unimplemeted!")
        }
    }
}

fn main() {
    println!("Welcome to Xnoe's Rust Calculator!");
    loop {
        let mut program = String::new();

        loop {
            let mut buffer = String::new();
            print!("> ");
            std::io::stdout().flush();
            std::io::stdin()
                .read_line(&mut buffer)
                .expect("Failed to read line!");
            
            match buffer.as_str().trim() {
                "" => break,
                s => program += s
            }
        }
        let tokens = tokenise((program.as_str().trim()).to_string());
        //print_tokens(&tokens);

        let ast = Parser::parse(&mut itertools::peek_nth(tokens.iter()));

        Interpreter::run(ast);
        //println!("Answer: {}", previous_answer);
    }
}
