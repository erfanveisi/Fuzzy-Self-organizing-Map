function [ y ] = tmf( x, l, c, r )

if x <= c
    if x <= l
        y = 0;
        return;
    else
        y = (x-l)/(c-l);
        return;
    end
else
    if x >= r
        y = 0;
        return;
    else
        y = (x-r)/(c-r);
        return;
    end
end

end

