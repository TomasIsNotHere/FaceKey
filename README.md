Aplikace FaceKey je maturitní projek, který se zabývá rozpoznáním člověka pomocí kamery v počítači/mobilu přes webové rozhraní. 

Aplikace běží na domácím webovém serveru kde se o chod webové části stará Flask. Webové rozhraní s pomocí kamery zařízení vezme snímky a ty následně předá k datové úpravě pro modely strojového učení. Tyto modely se následně starají o rozpoznání člověka na kameře a o odhalení Spoofingu.

Díky nedostatku dat nejsou modely dokonalé a dělají chyby při různorodosti vstupů. Dosazení aplikace na web je také spíše provizorní řešení díky deadlinu pro odevzdání. 

***Update*** 
Po analýze výsledku, jsem došel k názoru, že pro nasazení a odhalení spoofingu existují lepší aleternativy a proto pracuji na další verzi.
