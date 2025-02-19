grammar SQLPP;

expr: sql_expr ;
sql_expr: select_clause from_clause where_clause? group_by_clause? order_by_clause? limit_clause? ;
select_clause: 'SELECT' proj_list ;
proj_list: item | item ',' proj_list ;
from_clause: 'FROM' table_or_subquery ;
table_or_subquery: table_def | '(' subquery ')' ;
table_def: VAR ;
subquery: sql_expr | match_expr ;
where_clause: 'WHERE' predicates ;
predicates: predicate | predicate 'AND' predicates ;
predicate: item_or_sub_expr cmp_symbol item_or_sub_expr
         | func_expr
         ;
group_by_clause: 'GROUP' 'BY' var ;
order_by_clause: 'ORDER' 'BY' var order? ;
limit_clause: 'LIMIT' INT ;

match_expr: 'MATCH' obj_def traversals? 'RETURN' vars ;
obj_def: '{' kv_pairs '}' ;
kv_pairs: VAR ':' item_or_predicates
        | VAR ':' item_or_predicates ',' kv_pairs
        ;
item_or_predicates: item | '(' predicates ')' ;

traversals: traversal | traversal traversals ;
traversal: edge_def obj_def? ;
edge_def: edge_direction '(' STRING ')' ;
edge_direction: '.inE' | '.outE' | '.inV' | '.outV' ;

order: 'ASC' | 'DESC' ;
item_or_sub_expr: item | '(' subquery ')' ;
item: var
    | var 'AS' VAR
    | func_expr
    | func_expr 'AS' VAR
    | STAR
    | literal
    ;
vars: var | var ',' vars ;
var: VAR | VAR '.' VAR ;
func_expr: func_name '(' func_input ')' ;
func_name: VAR ;
func_input: item | item ',' func_input ;
cmp_symbol: '=' | '>' | '>=' | '<' | '<=' | '<>' | 'IN' | '@>' | 'IS' | 'IS NOT' | 'CONTAINS' ;
literal: INT | STRING | FLOAT | 'NULL' ;
VAR: [a-zA-Z_][a-zA-Z0-9_]* ;
STAR: '*' ;
INT: [0-9]+ ;
STRING: '"' ~('"')* '"' | '\'' ~('\'')* '\'' ;
FLOAT: INT '.' INT ;


WS: [ \t\r\n]+ -> skip;