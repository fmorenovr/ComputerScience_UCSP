function [V1, LAMBDAS1] = sort_eig(V,LAMBDAS)
    V1 = V;
    LAMBDAS1 = LAMBDAS;
    i = 1;
    initialVal = i;
    for i = 1:size(LAMBDAS1)(1)
        cur = LAMBDAS1(i,i);
        curV = V1(:,i);
        j = i-1;
        while ( ge(j,initialVal) && (LAMBDAS1(j,j) > cur))
            LAMBDAS1(j+1,j+1) = LAMBDAS1(j,j);
            V1(:,j+1) = V1(:,j);
            j = j-1;
        endwhile
        LAMBDAS1(j+1,j+1) = cur;
        LAMBDAS1 = fliplr(LAMBDAS1);
        LAMBDAS1 = flip(LAMBDAS1);
        V1(:,j+1) = curV;
        V1 = fliplr(V1);
    endfor
endfunction
