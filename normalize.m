
normalize_min_max=[0 1];

x_size=size(data);
x_normalized=zeros(x_size(1),x_size(2));

x_max=max(max(data));
x_min=min(min(data));
nesbat=abs(normalize_min_max(2)-normalize_min_max(1))/(abs(x_max-x_min));
for nn=1:x_size(1)
    for mm=1:x_size(2)
        x_normalized(nn,mm)=normalize_min_max(1)+(data(nn,mm)-x_min)*nesbat;
    end
end

data=x_normalized;
